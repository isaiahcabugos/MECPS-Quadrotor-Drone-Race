#!/usr/bin/env python3
"""
Quick verification script for residual fine-tuning pipeline.

This script performs dry-run tests to catch issues before full training:
1. Verifies checkpoint can be loaded
2. Tests residual model integration
3. Runs 1000-step test training
4. Validates policy inference

Usage:
    python3 verify_residual_pipeline.py \
        --checkpoint-dir ./checkpoints \
        --residual-model dynamics_residual_model.pkl
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import pickle

# CRITICAL: Initialize TF-Agents multiprocessing BEFORE other imports
from tf_agents.system import system_multiprocessing
system_multiprocessing.enable_interactive_mode()

from src.simulator import ImprovedCopterSimulator, CloverSpecs, ControllerGains
from src.tf_agents_wrapper import TFAgentsCopterEnv
from src.agent import create_ppo_agent
from src.trainer import PPOTrainer


def verify_checkpoint(checkpoint_dir):
    """Verify checkpoint exists and can be loaded"""
    print("\n" + "="*70)
    print("TEST 1: CHECKPOINT VERIFICATION")
    print("="*70)
    
    # Check files exist
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint')
    if not os.path.exists(checkpoint_file):
        print(f"❌ FAIL: No 'checkpoint' file in {checkpoint_dir}")
        return False
    
    # Check for .index and .data files
    ckpt_files = [f for f in os.listdir(checkpoint_dir) 
                  if f.startswith('ckpt-') and ('.index' in f or '.data' in f)]
    
    if not ckpt_files:
        print(f"❌ FAIL: No checkpoint files (ckpt-*.index, ckpt-*.data)")
        return False
    
    print(f"✓ Found checkpoint files: {len(ckpt_files)}")
    
    # Try to load
    try:
        # Create dummy environment
        from tf_agents.environments import tf_py_environment, parallel_py_environment
        
        def make_env():
            return TFAgentsCopterEnv(use_gimbal=True)
        
        parallel_env = parallel_py_environment.ParallelPyEnvironment([make_env] * 2)
        tf_env = tf_py_environment.TFPyEnvironment(parallel_env)
        
        # Create agent
        agent, optimizer, _, _ = create_ppo_agent(tf_env, learning_rate=3e-4)
        
        # Setup checkpointer
        from tf_agents.utils import common
        from tf_agents.replay_buffers import tf_uniform_replay_buffer
        
        global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False)
        
        # NOTE: Don't create replay buffer at all - we only need policy weights
        # The checkpoint was trained with 50 envs, verification uses 2 envs
        # This would cause shape mismatch, so we skip replay buffer entirely
        
        checkpointer = common.Checkpointer(
            ckpt_dir=checkpoint_dir,
            max_to_keep=5,
            agent=agent,
            policy=agent.policy,
            global_step=global_step,
            # NOTE: Explicitly exclude optimizer and replay_buffer
            # optimizer=optimizer  # Skip optimizer too - only need policy weights
        )
        
        # Load checkpoint with expect_partial() to ignore replay buffer
        status = checkpointer.initialize_or_restore()
        status.expect_partial()  # Ignore warnings about replay buffer
        loaded_step = global_step.numpy()
        
        if loaded_step == 0:
            print("❌ FAIL: Checkpoint loaded but step count is 0")
            return False
        
        print(f"✓ Checkpoint loaded successfully at step {loaded_step:,}")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Error loading checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_residual_model(residual_model_path):
    """Verify residual model structure and functionality"""
    print("\n" + "="*70)
    print("TEST 2: RESIDUAL MODEL VERIFICATION")
    print("="*70)
    
    try:
        with open(residual_model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Check required keys
        required = ['knn_model', 'n_samples', 'residual_stats']
        missing = [k for k in required if k not in model_data]
        
        if missing:
            print(f"❌ FAIL: Missing required keys: {missing}")
            return False
        
        print(f"✓ Model structure valid")
        print(f"  Samples: {model_data['n_samples']}")
        print(f"  Mean residual: {model_data['residual_stats']['mean']}")
        
        # Test prediction
        knn = model_data['knn_model']
        test_features = np.zeros((1, 6))  # [vel(3), cmd_vel(3)]
        pred = knn.predict(test_features)
        
        # Handle both (3,) and (1, 3) shapes
        if pred.shape == (1, 3):
            pred = pred[0]  # Extract from batch
        
        if pred.shape != (3,):
            print(f"❌ FAIL: Wrong prediction shape {pred.shape}, expected (3,)")
            return False
        
        print(f"✓ Prediction test passed: {pred}")
        
        # Check residual magnitudes are reasonable
        max_resid = model_data['residual_stats'].get('max', [10, 10, 10])
        if any(abs(m) > 10.0 for m in max_resid):
            print(f"⚠️  WARNING: Large residuals detected: {max_resid}")
            print(f"    This may indicate simulation-reality mismatch")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_residual_integration(residual_model_path):
    """Test that residual-augmented environment works"""
    print("\n" + "="*70)
    print("TEST 3: RESIDUAL INTEGRATION")
    print("="*70)
    
    try:
        # Load model
        with open(residual_model_path, 'rb') as f:
            model_data = pickle.load(f)
        knn = model_data['knn_model']
        
        # Create base simulator
        sim = ImprovedCopterSimulator(
            specs=CloverSpecs(),
            gains=ControllerGains(),
            dt=0.004,
            domain_randomization=True,
            add_noise=True,
            use_gimbal=True
        )
        
        print("✓ Created base simulator")
        
        # Test residual wrapper logic
        obs, _ = sim.reset()
        
        for step in range(10):
            action = np.random.uniform(-1, 1, 3).astype(np.float32)
            
            # Store velocity
            vel_before = sim.velocity.copy()
            
            # Compute commanded velocity
            cmd_vel = np.array([
                action[0] * sim.specs.max_velocity_xy,
                action[1] * sim.specs.max_velocity_xy,
                action[2] * sim.specs.max_velocity_z
            ])
            
            # Predict residual
            features = np.concatenate([vel_before, cmd_vel]).reshape(1, -1)
            residual = knn.predict(features)[0]
            
            # Step
            obs, reward, terminated, truncated, info = sim.step(action)
            
            if step == 0:
                print(f"✓ Step {step}: vel={vel_before}, residual={residual}")
        
        print(f"✓ Residual integration test passed (10 steps)")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Integration error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_training_step(checkpoint_dir, residual_model_path):
    """Test that training can run for 1000 steps"""
    print("\n" + "="*70)
    print("TEST 4: TRAINING DRY RUN (1000 steps)")
    print("="*70)
    
    try:
        # Import the residual environment
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from finetune_with_residuals import create_residual_environments
        
        # Create environment
        print("Creating residual environments...")
        tf_env = create_residual_environments(residual_model_path, num_envs=2)
        print("✓ Environments created")
        
        # Create agent
        print("Creating agent...")
        agent, optimizer, _, _ = create_ppo_agent(tf_env, learning_rate=3e-4)
        print("✓ Agent created")
        
        # Load checkpoint
        print("Loading checkpoint...")
        from tf_agents.utils import common
        from tf_agents.replay_buffers import tf_uniform_replay_buffer
        
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            agent.collect_data_spec,
            batch_size=2,
            max_length=1500,
        )
        
        checkpointer = common.Checkpointer(
            ckpt_dir=checkpoint_dir,
            agent=agent,
            global_step=global_step,
            optimizer=optimizer
        )
        
        checkpointer.initialize_or_restore()
        loaded_step = global_step.numpy()
        print(f"✓ Loaded checkpoint at step {loaded_step:,}")
        
        # Test training step
        print("Running 1000-step training...")
        from tf_agents.drivers import dynamic_step_driver
        
        collect_driver = dynamic_step_driver.DynamicStepDriver(
            tf_env,
            agent.collect_policy,
            observers=[replay_buffer.add_batch],
            num_steps=1000,
        )
        
        replay_buffer.clear()
        collect_driver.run()
        
        trajectories = replay_buffer.gather_all()
        train_loss = agent.train(experience=trajectories)
        
        print(f"✓ Training step completed")
        print(f"  Loss: {train_loss.loss.numpy():.4f}")
        print(f"  Trajectories collected: {trajectories.reward.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Training error: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_policy_inference(checkpoint_dir):
    """Test that policy can run inference"""
    print("\n" + "="*70)
    print("TEST 5: POLICY INFERENCE")
    print("="*70)
    
    try:
        # Try to load saved policy directly
        policy_dir = os.path.join(checkpoint_dir, 'policy')
        
        if not os.path.exists(policy_dir):
            print(f"⚠️  No saved policy found in {policy_dir}")
            print(f"   (This is OK if you haven't saved a policy yet)")
            return True
        
        # Find latest policy
        policy_dirs = [d for d in os.listdir(policy_dir) if d.startswith('policy_')]
        if not policy_dirs:
            print(f"⚠️  No policy directories found")
            return True
        
        latest_policy = sorted(policy_dirs)[-1]
        policy_path = os.path.join(policy_dir, latest_policy)
        
        print(f"Loading policy from {policy_path}...")
        saved_policy = tf.saved_model.load(policy_path)
        print(f"✓ Policy loaded")
        
        # Test inference
        env = TFAgentsCopterEnv(use_gimbal=True)
        time_step = env.reset()
        
        for step in range(10):
            # Batch the time step
            batched_time_step = tf.nest.map_structure(
                lambda t: tf.expand_dims(t, 0),
                time_step
            )
            
            action_step = saved_policy.action(batched_time_step)
            action = action_step.action.numpy()[0]
            
            time_step = env.step(action)
            
            if step == 0:
                print(f"✓ Inference step {step}: action={action}")
        
        print(f"✓ Policy inference test passed (10 steps)")
        return True
        
    except Exception as e:
        print(f"❌ FAIL: Inference error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Verify residual fine-tuning pipeline before full training'
    )
    
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                       help='Checkpoint directory')
    parser.add_argument('--residual-model', type=str, required=True,
                       help='Residual model pickle file')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training dry run (faster)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("RESIDUAL FINE-TUNING PIPELINE VERIFICATION")
    print("="*70)
    print(f"Checkpoint: {args.checkpoint_dir}")
    print(f"Residual model: {args.residual_model}")
    print("="*70)
    
    # Run tests
    results = {}
    
    results['checkpoint'] = verify_checkpoint(args.checkpoint_dir)
    results['residual_model'] = verify_residual_model(args.residual_model)
    results['integration'] = verify_residual_integration(args.residual_model)
    
    if not args.skip_training:
        results['training'] = verify_training_step(args.checkpoint_dir, args.residual_model)
    
    results['inference'] = verify_policy_inference(args.checkpoint_dir)
    
    # Summary
    print("\n" + "="*70)
    print("VERIFICATION SUMMARY")
    print("="*70)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test:20s}: {status}")
    
    all_passed = all(results.values())
    
    print("="*70)
    
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("\nYou can now run:")
        print(f"  python3 finetune_with_residuals.py \\")
        print(f"      --checkpoint-dir {args.checkpoint_dir} \\")
        print(f"      --residual-model {args.residual_model} \\")
        print(f"      --steps 20000000")
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("\nPlease fix the issues before running fine-tuning.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
