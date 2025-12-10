#!/usr/bin/env python3
"""
Fine-tune 100M checkpoint with residual-augmented dynamics.

This script:
1. Loads your 100M-step trained policy
2. Wraps the simulator with learned dynamics residuals
3. Continues training for 10M more steps (→ 110M total)
4. Saves fine-tuned checkpoint

Usage:
    python3 finetune_with_residuals.py \
        --checkpoint-dir ./drone_training_checkpoints \
        --residual-model dynamics_residual_model.pkl \
        --steps 10000000
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime

# TF-Agents imports
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.environments import tf_py_environment, parallel_py_environment
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.policies import policy_saver
from tf_agents.specs import tensor_spec

# Import your simulator (adjust path as needed)
try:
    # If running from same directory as lightweightdrltrainer_v6.py
    from lightweightdrltrainer_v6 import (
        ImprovedCopterSimulator, 
        CloverSpecs, 
        ControllerGains,
        CopterTFPyEnvironment
    )
except ImportError:
    print("ERROR: Could not import from lightweightdrltrainer_v6.py")
    print("Make sure this script is in the same directory or adjust import path")
    sys.exit(1)

# Import residual wrapper
import pickle


class ResidualAugmentedSimulator:
    """Wraps ImprovedCopterSimulator with dynamics residuals"""
    
    def __init__(self, base_sim, residual_model_path, apply_y_flip=True):
        self.base_sim = base_sim
        self.apply_y_flip = apply_y_flip
        
        # Load residual model
        with open(residual_model_path, 'rb') as f:
            model_data = pickle.load(f)
        self.knn_model = model_data['knn_model']
        self.residual_stats = model_data.get('residual_stats', {})
        
        print(f"✓ Loaded residual model:")
        print(f"  Samples: {model_data.get('n_samples')}")
        print(f"  Mean residual: {self.residual_stats.get('mean')}")
        
        # Copy attributes from base sim
        self.observation_space = base_sim.observation_space
        self.action_space = base_sim.action_space
    
    def reset(self, **kwargs):
        return self.base_sim.reset(**kwargs)
    
    def step(self, action):
        # Store current velocity
        current_velocity = self.base_sim.velocity.copy()
        
        # Compute commanded velocity with Y-flip
        if self.apply_y_flip:
            action_flipped = action * np.array([1.0, -1.0, 1.0], dtype=np.float32)
        else:
            action_flipped = action
        
        cmd_velocity = action_flipped * np.array([5.0, 5.0, 2.0], dtype=np.float32)
        
        # Execute base step
        obs, reward, terminated, truncated, info = self.base_sim.step(action)
        
        # Apply residual
        features = np.concatenate([current_velocity, cmd_velocity]).reshape(1, -1)
        residual_accel = self.knn_model.predict(features)[0]
        
        self.base_sim.velocity += residual_accel * self.base_sim.dt
        self.base_sim.position += self.base_sim.velocity * self.base_sim.dt
        
        obs = self.base_sim._get_observation()
        info['residual_mag'] = np.linalg.norm(residual_accel)
        
        return obs, reward, terminated, truncated, info
    
    def __getattr__(self, name):
        return getattr(self.base_sim, name)


def create_residual_tf_environment(residual_model_path, num_envs=32, seed=None):
    """Create parallel TF environments with residual augmentation"""
    
    def make_env(seed_offset=0):
        def _init():
            # Create base simulator
            base_sim = ImprovedCopterSimulator(
                specs=CloverSpecs(),
                gains=ControllerGains(),
                domain_randomization=True,
                add_noise=True,
                randomize_track=True,
                num_gates=7
            )
            
            # Wrap with residuals
            augmented_sim = ResidualAugmentedSimulator(
                base_sim=base_sim,
                residual_model_path=residual_model_path,
                apply_y_flip=True
            )
            
            return augmented_sim
        
        return _init
    
    # Create parallel environments
    py_envs = [make_env(i) for i in range(num_envs)]
    
    # Wrap each in TF-PyEnvironment wrapper
    # Note: CopterTFPyEnvironment expects to create its own simulator
    # We need to modify this to accept an existing simulator
    
    # Instead, we'll create environments directly
    from tf_agents.environments import py_environment
    from tf_agents.specs import array_spec
    
    class ResidualCopterEnv(py_environment.PyEnvironment):
        def __init__(self, env_creator):
            super().__init__()
            self._env = env_creator()
            self._observation_spec = array_spec.ArraySpec(
                shape=(30,), dtype=np.float32, name='observation')
            self._action_spec = array_spec.BoundedArraySpec(
                shape=(3,), dtype=np.float32, minimum=-1.0, maximum=1.0, name='action')
            self._episode_ended = False
        
        def observation_spec(self):
            return self._observation_spec
        
        def action_spec(self):
            return self._action_spec
        
        def _reset(self):
            self._episode_ended = False
            obs, info = self._env.reset()
            return tf_agents.trajectories.time_step.restart(obs)
        
        def _step(self, action):
            if self._episode_ended:
                return self.reset()
            
            obs, reward, terminated, truncated, info = self._env.step(action)
            self._episode_ended = terminated or truncated
            
            if self._episode_ended:
                return tf_agents.trajectories.time_step.termination(obs, reward)
            else:
                return tf_agents.trajectories.time_step.transition(obs, reward)
    
    # Create wrapped environments
    py_envs_wrapped = [ResidualCopterEnv(env_creator) for env_creator in py_envs]
    
    # Convert to TF environments
    tf_env = tf_py_environment.TFPyEnvironment(
        parallel_py_environment.ParallelPyEnvironment(py_envs_wrapped)
    )
    
    return tf_env


def load_checkpoint_and_continue(checkpoint_dir, residual_model_path, 
                                 additional_steps=10_000_000, num_envs=32):
    """
    Load 100M checkpoint and continue training with residuals.
    """
    
    print("="*70)
    print("FINE-TUNING WITH RESIDUAL-AUGMENTED DYNAMICS")
    print("="*70)
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Residual model: {residual_model_path}")
    print(f"Additional steps: {additional_steps:,}")
    print(f"Num parallel envs: {num_envs}")
    print("="*70)
    
    # Create environment with residuals
    print("\n📦 Creating residual-augmented environments...")
    tf_env = create_residual_tf_environment(
        residual_model_path=residual_model_path,
        num_envs=num_envs
    )
    print(f"✓ Created {num_envs} parallel environments")
    
    # Recreate agent architecture (must match training)
    print("\n🏗️  Recreating agent architecture...")
    
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        fc_layer_params=(128, 128),
        activation_fn=tf.nn.leaky_relu
    )
    
    value_net = value_network.ValueNetwork(
        tf_env.observation_spec(),
        fc_layer_params=(128, 128),
        activation_fn=tf.nn.leaky_relu
    )
    
    # Create optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
    
    # Create agent
    global_step = tf.Variable(0, dtype=tf.int64, name='global_step')
    
    agent = ppo_clip_agent.PPOClipAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        optimizer=optimizer,
        actor_net=actor_net,
        value_net=value_net,
        num_epochs=10,
        train_step_counter=global_step,
        normalize_observations=False,
        normalize_rewards=False,
        use_gae=True,
        use_td_lambda_return=True,
        lambda_value=0.95,
        discount_factor=0.99,
        gradient_clipping=0.5,
        value_pred_loss_coef=0.5,
        entropy_regularization=0.01,
        importance_ratio_clipping=0.2,
        log_prob_clipping=0.0,
        kl_cutoff_factor=2.0,
        adaptive_kl_target=0.01,
        adaptive_kl_tolerance=0.3,
    )
    
    agent.initialize()
    print("✓ Agent created")
    
    # Setup checkpoint restoration
    print(f"\n📂 Loading checkpoint from {checkpoint_dir}...")
    
    train_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        agent=agent,
        global_step=global_step,
        optimizer=optimizer
    )
    
    # Load checkpoint
    train_checkpointer.initialize_or_restore()
    current_step = global_step.numpy()
    
    if current_step == 0:
        print("❌ ERROR: No checkpoint found or failed to load!")
        print(f"   Check that {checkpoint_dir} contains valid checkpoints")
        return
    
    print(f"✓ Loaded checkpoint at step {current_step:,}")
    print(f"   Will train to step {current_step + additional_steps:,}")
    
    # Setup replay buffer
    print("\n💾 Setting up replay buffer...")
    
    replay_buffer_capacity = 1500 * num_envs  # episode_length * num_envs
    
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        agent.collect_data_spec,
        batch_size=num_envs,
        max_length=replay_buffer_capacity
    )
    print(f"✓ Replay buffer capacity: {replay_buffer_capacity}")
    
    # Setup policy saver
    print("\n💾 Setting up policy saver...")
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)
    
    # Training loop
    print("\n🚀 Starting fine-tuning...")
    print("="*70)
    
    target_step = current_step + additional_steps
    checkpoint_freq = 1_000_000  # Save every 1M steps
    last_checkpoint_step = current_step
    
    # Collect initial data
    print(f"\n📊 Collecting initial trajectories...")
    tf_env.reset()
    
    while current_step < target_step:
        # Collect trajectories
        time_step = tf_env.current_time_step()
        
        for _ in range(1500):  # One episode
            action_step = agent.policy.action(time_step)
            next_time_step = tf_env.step(action_step.action)
            
            traj = tf_agents.trajectories.from_transition(
                time_step, action_step, next_time_step
            )
            replay_buffer.add_batch(traj)
            
            time_step = next_time_step
            
            if time_step.is_last():
                break
        
        # Train on collected data
        dataset = replay_buffer.as_dataset(
            sample_batch_size=num_envs,
            num_steps=2
        )
        iterator = iter(dataset)
        
        experience, _ = next(iterator)
        train_loss = agent.train(experience)
        
        # Increment step
        current_step = global_step.numpy()
        
        # Logging
        if current_step % 1000 == 0:
            avg_return = tf.reduce_mean(experience.reward).numpy()
            print(f"Step {current_step:,}/{target_step:,} | "
                  f"Loss: {train_loss.loss.numpy():.4f} | "
                  f"Avg Reward: {avg_return:.3f}")
        
        # Checkpoint saving
        if current_step - last_checkpoint_step >= checkpoint_freq:
            print(f"\n💾 Saving checkpoint at step {current_step:,}...")
            train_checkpointer.save(global_step=current_step)
            
            policy_dir = os.path.join(checkpoint_dir, f'policy_finetuned_{current_step}')
            tf_policy_saver.save(policy_dir)
            print(f"✓ Saved to {policy_dir}")
            
            last_checkpoint_step = current_step
        
        # Clear buffer periodically
        if current_step % 10000 == 0:
            replay_buffer.clear()
    
    # Final save
    print(f"\n💾 Saving final checkpoint at step {current_step:,}...")
    train_checkpointer.save(global_step=current_step)
    
    policy_dir = os.path.join(checkpoint_dir, f'policy_finetuned_final_{current_step}')
    tf_policy_saver.save(policy_dir)
    print(f"✓ Final policy saved to {policy_dir}")
    
    print("\n" + "="*70)
    print("✅ FINE-TUNING COMPLETE!")
    print("="*70)
    print(f"Final step: {current_step:,}")
    print(f"Policy saved to: {policy_dir}")


def main():
    parser = argparse.ArgumentParser(description='Fine-tune with residuals')
    parser.add_argument('--checkpoint-dir', type=str, 
                       default='./drone_training_checkpoints',
                       help='Directory containing 100M checkpoint')
    parser.add_argument('--residual-model', type=str,
                       required=True,
                       help='Path to dynamics_residual_model.pkl')
    parser.add_argument('--steps', type=int, default=10_000_000,
                       help='Additional training steps (default: 10M)')
    parser.add_argument('--num-envs', type=int, default=32,
                       help='Number of parallel environments')
    
    args = parser.parse_args()
    
    # Verify files exist
    if not os.path.exists(args.checkpoint_dir):
        print(f"❌ ERROR: Checkpoint directory not found: {args.checkpoint_dir}")
        return
    
    if not os.path.exists(args.residual_model):
        print(f"❌ ERROR: Residual model not found: {args.residual_model}")
        return
    
    # Run fine-tuning
    load_checkpoint_and_continue(
        checkpoint_dir=args.checkpoint_dir,
        residual_model_path=args.residual_model,
        additional_steps=args.steps,
        num_envs=args.num_envs
    )


if __name__ == '__main__':
    main()
