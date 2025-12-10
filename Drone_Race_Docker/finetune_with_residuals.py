#!/usr/bin/env python3
"""
Fine-tune 100M checkpoint with residual-augmented dynamics.

This script:
1. Loads your 100M-step trained policy
2. Wraps the simulator with learned dynamics residuals
3. Continues training for 20M more steps (→ 120M total)
4. Saves fine-tuned checkpoint

Usage:
    python3 finetune_with_residuals.py \
        --checkpoint-dir ./checkpoints \
        --residual-model dynamics_residual_model.pkl \
        --steps 20000000
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
import pickle

# CRITICAL: Initialize TF-Agents multiprocessing BEFORE other imports
# Only initialize if not already done (to avoid double-init in verification)
from tf_agents.system import system_multiprocessing
try:
    system_multiprocessing.enable_interactive_mode()
except ValueError:
    # Already initialized - that's OK
    pass

# TF-Agents imports
from tf_agents.environments import tf_py_environment, parallel_py_environment
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.utils import common
from tf_agents.policies import policy_saver
from tf_agents.drivers import dynamic_step_driver

# Import your modules
from src.simulator import ImprovedCopterSimulator, CloverSpecs, ControllerGains
from src.tf_agents_wrapper import TFAgentsCopterEnv
from src.agent import create_ppo_agent
from src.trainer import PPOTrainer


class ResidualAugmentedSimulator:
    """Wraps ImprovedCopterSimulator with dynamics residuals"""
    
    def __init__(self, base_sim, residual_model_path):
        """
        Args:
            base_sim: ImprovedCopterSimulator instance
            residual_model_path: Path to pickled residual model
        """
        self.base_sim = base_sim
        
        # Load residual model
        print(f"Loading residual model from {residual_model_path}")
        with open(residual_model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.knn_model = model_data['knn_model']
        self.residual_stats = model_data.get('residual_stats', {})
        
        print(f"✓ Loaded residual model:")
        print(f"  Samples: {model_data.get('n_samples', 'unknown')}")
        print(f"  Mean residual: {self.residual_stats.get('mean', 'unknown')}")
        
        # Copy attributes from base sim
        self.observation_space = base_sim.observation_space
        self.action_space = base_sim.action_space
        
        # Track residual magnitudes for logging
        self.residual_history = []
    
    def reset(self, **kwargs):
        """Reset the environment"""
        self.residual_history = []
        return self.base_sim.reset(**kwargs)
    
    def step(self, action):
        """
        Execute step with residual dynamics correction.
        
        The residual model predicts acceleration corrections based on:
        - Current velocity (body frame)
        - Commanded velocity (from action)
        """
        # Store current state
        current_velocity = self.base_sim.velocity.copy()
        
        # Execute base simulator step
        obs, reward, terminated, truncated, info = self.base_sim.step(action)
        
        # Compute commanded velocity from action
        # Actions are normalized [-1, 1], map to velocity ranges
        cmd_velocity = np.array([
            action[0] * self.base_sim.specs.max_velocity_xy,  # vx
            action[1] * self.base_sim.specs.max_velocity_xy,  # vy
            action[2] * self.base_sim.specs.max_velocity_z    # vz
        ], dtype=np.float32)
        
        # Predict residual acceleration
        features = np.concatenate([current_velocity, cmd_velocity]).reshape(1, -1)
        residual_accel = self.knn_model.predict(features)[0]  # Extract from batch (1, 3) -> (3,)
        
        # Apply residual correction to velocity
        dt = self.base_sim.control_dt  # Use control timestep
        self.base_sim.velocity += residual_accel * dt
        
        # Update position with corrected velocity
        self.base_sim.position += self.base_sim.velocity * dt
        
        # Get updated observation
        obs = self.base_sim._get_observation()
        
        # Track residual magnitude
        residual_mag = np.linalg.norm(residual_accel)
        self.residual_history.append(residual_mag)
        info['residual_mag'] = residual_mag
        info['avg_residual_mag'] = np.mean(self.residual_history)
        
        return obs, reward, terminated, truncated, info
    
    def __getattr__(self, name):
        """Forward attribute access to base simulator"""
        return getattr(self.base_sim, name)


class ResidualAugmentedTFEnv(TFAgentsCopterEnv):
    """TF-Agents wrapper with residual dynamics"""
    
    def __init__(self, residual_model_path, seed=None, episode_length=1500, use_gimbal=True):
        """
        Create environment with residual dynamics.
        
        Args:
            residual_model_path: Path to pickled residual model
            seed: Random seed
            episode_length: Max steps per episode
            use_gimbal: Use gimbal-stabilized camera
        """
        # Create base simulator
        base_sim = ImprovedCopterSimulator(
            specs=CloverSpecs(),
            gains=ControllerGains(),
            dt=0.004,
            domain_randomization=True,
            add_noise=True,  # Enable noise for realism
            use_gimbal=use_gimbal
        )
        
        # Wrap with residuals
        self._sim = ResidualAugmentedSimulator(
            base_sim=base_sim,
            residual_model_path=residual_model_path
        )
        
        # Initialize parent class attributes
        self.episode_length = episode_length
        
        if seed is not None:
            self._sim.reset(seed=seed)
        
        # Set specs (copied from TFAgentsCopterEnv)
        from tf_agents.specs import array_spec
        
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(3,),
            dtype=np.float32,
            minimum=-1.0,
            maximum=1.0,
            name='action'
        )
        
        self._observation_spec = array_spec.ArraySpec(
            shape=(30,),
            dtype=np.float32,
            name='observation'
        )
        
        self._episode_ended = False
        self._episode_step = 0


def create_residual_environments(residual_model_path, num_envs=20):
    """
    Create parallel environments with residual augmentation.
    
    Args:
        residual_model_path: Path to residual model pickle
        num_envs: Number of parallel environments
    
    Returns:
        TF-Agents parallel environment
    """
    print(f"\n📦 Creating {num_envs} residual-augmented environments...")
    
    def make_env(seed_offset=0):
        """Environment factory"""
        return ResidualAugmentedTFEnv(
            residual_model_path=residual_model_path,
            seed=42 + seed_offset,
            episode_length=1500,
            use_gimbal=True
        )
    
    # Create parallel environments
    parallel_env = parallel_py_environment.ParallelPyEnvironment(
        [lambda i=i: make_env(i) for i in range(num_envs)]
    )
    
    # Wrap in TF environment
    tf_env = tf_py_environment.TFPyEnvironment(parallel_env)
    
    print(f"✓ Created {num_envs} parallel environments with residual dynamics")
    return tf_env


def load_checkpoint_and_finetune(
    checkpoint_dir,
    residual_model_path,
    additional_steps=20_000_000,
    num_envs=20,
    checkpoint_freq=1_000_000
):
    """
    Load checkpoint and continue training with residuals.
    
    Args:
        checkpoint_dir: Directory containing checkpoint
        residual_model_path: Path to residual model pickle
        additional_steps: Additional training steps
        num_envs: Number of parallel environments
        checkpoint_freq: Checkpoint frequency
    """
    
    print("="*70)
    print("FINE-TUNING WITH RESIDUAL-AUGMENTED DYNAMICS")
    print("="*70)
    print(f"Checkpoint dir: {checkpoint_dir}")
    print(f"Residual model: {residual_model_path}")
    print(f"Additional steps: {additional_steps:,}")
    print(f"Num parallel envs: {num_envs}")
    print(f"Checkpoint frequency: {checkpoint_freq:,}")
    print("="*70)
    
    # Create environment with residuals
    tf_env = create_residual_environments(
        residual_model_path=residual_model_path,
        num_envs=num_envs
    )
    
    # Create agent (matches agent.py architecture)
    print("\n🗃️  Creating PPO agent...")
    agent, optimizer, actor_net, value_net = create_ppo_agent(
        tf_env=tf_env,
        learning_rate=3e-4
    )
    print("✓ PPO agent created")
    
    # Setup trainer
    print("\n💾 Setting up trainer...")
    # NOTE: Log directory MUST be writable, checkpoints dir is read-only
    log_dir = '/workspace/outputs/logs_finetuned'
    trainer = PPOTrainer(
        agent=agent,
        tf_env=tf_env,
        optimizer=optimizer,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        checkpoint_freq=checkpoint_freq
    )
    
    # Setup checkpointing (this creates replay buffer internally)
    print("\n📂 Loading checkpoint...")
    
    # Manually setup replay buffer first
    n_steps_per_env = 1500  # Episode length
    trainer.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        agent.collect_data_spec,
        batch_size=tf_env.batch_size,
        max_length=n_steps_per_env,
    )
    
    # Now setup checkpointer WITHOUT replay buffer for loading
    # (trainer._setup_checkpointing() would include it, causing shape mismatch)
    print(f"   Note: Skipping replay buffer restoration (env count may differ from checkpoint)")
    print(f"   Your checkpoint: trained with different num_envs")
    print(f"   Fine-tuning with: {num_envs} envs")
    
    # Create separate checkpointer for loading (no replay buffer)
    load_checkpointer = common.Checkpointer(
        ckpt_dir=checkpoint_dir,
        max_to_keep=5,
        agent=agent,
        policy=agent.policy,
        global_step=trainer.global_step,
        optimizer=optimizer
        # NOTE: replay_buffer excluded - would cause shape mismatch
    )
    
    # Load checkpoint
    status = load_checkpointer.initialize_or_restore()
    status.expect_partial()  # Ignore warnings about missing replay buffer
    loaded_step = trainer.global_step.numpy()
    agent.train_step_counter.assign(loaded_step)
    
    # Now setup full checkpointer for saving (includes replay buffer)
    trainer._setup_checkpointing()
    
    if loaded_step == 0:
        print("❌ ERROR: No checkpoint found or failed to load!")
        print(f"   Check that {checkpoint_dir} contains valid checkpoints")
        return
    
    print(f"✓ Loaded checkpoint at step {loaded_step:,}")
    print(f"   Will train to step {loaded_step + additional_steps:,}")
    
    # Continue training
    print("\n🚀 Starting fine-tuning...")
    print("="*70)
    
    try:
        trainer.train(
            total_timesteps=loaded_step + additional_steps,
            n_steps_per_env=n_steps_per_env
        )
    except KeyboardInterrupt:
        print("\n⚠️ Fine-tuning interrupted")
        current = trainer.global_step.numpy()
        trainer.train_checkpointer.save(global_step=current)
        print(f"✓ Checkpoint saved at step {current:,}")
    
    print("\n" + "="*70)
    print("✅ FINE-TUNING COMPLETE!")
    print("="*70)
    print(f"Final step: {trainer.global_step.numpy():,}")
    print(f"Policy saved in: {checkpoint_dir}/policy/")
    print("="*70)


def verify_residual_model(residual_model_path):
    """
    Verify residual model can be loaded and has expected structure.
    
    Args:
        residual_model_path: Path to pickle file
    
    Returns:
        bool: True if valid
    """
    print(f"\n🔍 Verifying residual model: {residual_model_path}")
    
    try:
        with open(residual_model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Check required keys
        required_keys = ['knn_model', 'n_samples']
        for key in required_keys:
            if key not in model_data:
                print(f"❌ Missing required key: {key}")
                return False
        
        # Check KNN model
        knn = model_data['knn_model']
        if not hasattr(knn, 'predict'):
            print("❌ KNN model doesn't have predict method")
            return False
        
        # Test prediction
        test_features = np.zeros((1, 6))  # [vel(3), cmd_vel(3)]
        try:
            pred = knn.predict(test_features)
            
            # Handle sklearn KNN returning (1, 3) for single sample
            if pred.shape == (1, 3):
                pred = pred[0]  # Extract to (3,)
            
            if pred.shape != (3,):
                print(f"❌ Unexpected prediction shape: {pred.shape}, expected (3,)")
                return False
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return False
        
        print("✓ Residual model verification passed")
        print(f"  Samples: {model_data.get('n_samples')}")
        print(f"  Features: velocity(3) + cmd_velocity(3) = 6D")
        print(f"  Output: residual_acceleration(3)")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to load residual model: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune drone racing policy with residual dynamics'
    )
    
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='./checkpoints',
        help='Directory containing trained checkpoint (default: ./checkpoints)'
    )
    
    parser.add_argument(
        '--residual-model',
        type=str,
        required=True,
        help='Path to dynamics_residual_model.pkl (REQUIRED)'
    )
    
    parser.add_argument(
        '--steps',
        type=int,
        default=20_000_000,
        help='Additional training steps (default: 20M)'
    )
    
    parser.add_argument(
        '--num-envs',
        type=int,
        default=20,
        help='Number of parallel environments (default: 20)'
    )
    
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=1_000_000,
        help='Checkpoint frequency in steps (default: 1M)'
    )
    
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify residual model without training'
    )
    
    args = parser.parse_args()
    
    # Verify files exist
    if not os.path.exists(args.checkpoint_dir):
        print(f"❌ ERROR: Checkpoint directory not found: {args.checkpoint_dir}")
        sys.exit(1)
    
    if not os.path.exists(args.residual_model):
        print(f"❌ ERROR: Residual model not found: {args.residual_model}")
        sys.exit(1)
    
    # Verify residual model
    if not verify_residual_model(args.residual_model):
        print("\n❌ Residual model verification failed!")
        print("Please check your residual model file.")
        sys.exit(1)
    
    if args.verify_only:
        print("\n✓ Verification complete. Exiting (--verify-only).")
        return
    
    # Run fine-tuning
    load_checkpoint_and_finetune(
        checkpoint_dir=args.checkpoint_dir,
        residual_model_path=args.residual_model,
        additional_steps=args.steps,
        num_envs=args.num_envs,
        checkpoint_freq=args.checkpoint_freq
    )


if __name__ == '__main__':
    main()
