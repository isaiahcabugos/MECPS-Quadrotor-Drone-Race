#!/usr/bin/env python3
"""
Helper script to collect and process residual data from Gazebo.

This script:
1. Deploys trained policy in Gazebo
2. Records ground truth + VIO data
3. Fits dynamics residual model
4. Saves residual model for fine-tuning

Usage:
    # Step 1: Collect data
    python3 collect_residuals.py collect \
        --policy ./checkpoints/policy/policy_step_100000000 \
        --output gazebo_data.pkl \
        --duration 60

    # Step 2: Fit residual model
    python3 collect_residuals.py fit \
        --input gazebo_data.pkl \
        --output dynamics_residual_model.pkl
"""

import argparse
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from datetime import datetime
import matplotlib.pyplot as plt

# CRITICAL: Initialize TF-Agents multiprocessing for compatibility
from tf_agents.system import system_multiprocessing
system_multiprocessing.enable_interactive_mode()

# Your imports
from src.simulator import ImprovedCopterSimulator, CloverSpecs, ControllerGains
from src.tf_agents_wrapper import TFAgentsCopterEnv


def collect_data(policy_path, output_path, duration=60, use_mocap=True):
    """
    Collect data by running policy in Gazebo.
    
    Args:
        policy_path: Path to saved policy
        output_path: Where to save collected data
        duration: Flight duration in seconds
        use_mocap: Use motion capture for ground truth
    """
    import tensorflow as tf
    
    print("="*70)
    print("DATA COLLECTION FOR RESIDUAL LEARNING")
    print("="*70)
    print(f"Policy: {policy_path}")
    print(f"Duration: {duration}s (~{duration/20:.0f} laps)")
    print(f"Motion capture: {'enabled' if use_mocap else 'disabled'}")
    print("="*70)
    
    # Load policy
    print("\nLoading policy...")
    saved_policy = tf.saved_model.load(policy_path)
    print("✓ Policy loaded")
    
    # Create environment
    print("\nCreating environment...")
    env = TFAgentsCopterEnv(use_gimbal=True, episode_length=1500)
    
    # Create simulator reference for ground truth
    sim = env._sim
    
    print("✓ Environment created")
    print(f"  Position: {sim.position}")
    print(f"  Gates: {sim.num_gates}")
    
    # Data storage
    data = {
        'gt_position': [],
        'gt_velocity': [],
        'gt_orientation': [],
        'vio_position': [],  # In real Gazebo, replace with actual VIO
        'vio_velocity': [],
        'cmd_velocity': [],
        'actions': [],
        'rewards': [],
        'timestamps': []
    }
    
    # Reset environment
    time_step = env.reset()
    
    print("\n🚁 Starting data collection...")
    print("="*70)
    
    start_time = 0
    step = 0
    max_steps = int(duration / sim.control_dt)  # duration / 0.04
    
    while step < max_steps and not time_step.is_last():
        # Get action from policy
        batched_time_step = tf.nest.map_structure(
            lambda t: tf.expand_dims(t, 0),
            time_step
        )
        action_step = saved_policy.action(batched_time_step)
        action = action_step.action.numpy()[0]
        
        # Compute commanded velocity from action
        cmd_vel = np.array([
            action[0] * sim.specs.max_velocity_xy,
            action[1] * sim.specs.max_velocity_xy,
            action[2] * sim.specs.max_velocity_z
        ])
        
        # Store data BEFORE stepping
        data['gt_position'].append(sim.position.copy())
        data['gt_velocity'].append(sim.velocity.copy())
        data['gt_orientation'].append(sim.orientation.copy())
        
        # VIO estimate (for now, add noise to ground truth)
        # In real Gazebo, replace with actual VIO readings
        vio_pos = sim.position + np.random.normal(0, 0.05, 3)
        vio_vel = sim.velocity + np.random.normal(0, 0.1, 3)
        
        data['vio_position'].append(vio_pos)
        data['vio_velocity'].append(vio_vel)
        data['cmd_velocity'].append(cmd_vel)
        data['actions'].append(action)
        data['timestamps'].append(step * sim.control_dt)
        
        # Step environment
        time_step = env.step(action)
        reward = float(time_step.reward.numpy())
        data['rewards'].append(reward)
        
        # Progress logging
        if step % 250 == 0:
            elapsed = step * sim.control_dt
            print(f"Step {step:5d} | Time: {elapsed:5.1f}s | "
                  f"Pos: [{sim.position[0]:6.2f}, {sim.position[1]:6.2f}, {sim.position[2]:6.2f}] | "
                  f"Gates: {sim.gates_passed}")
        
        step += 1
    
    # Convert to numpy arrays
    for key in data:
        data[key] = np.array(data[key])
    
    print("\n" + "="*70)
    print("✓ Data collection complete!")
    print(f"  Steps: {len(data['gt_position'])}")
    print(f"  Duration: {data['timestamps'][-1]:.1f}s")
    print(f"  Gates passed: {sim.gates_passed}")
    print("="*70)
    
    # Save
    print(f"\n💾 Saving to {output_path}...")
    metadata = {
        'policy_path': policy_path,
        'duration': duration,
        'steps': step,
        'gates_passed': sim.gates_passed,
        'collection_date': datetime.now().isoformat(),
        'control_dt': sim.control_dt
    }
    
    output_data = {
        'data': data,
        'metadata': metadata
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"✓ Saved {len(data['gt_position'])} samples")
    print(f"  File size: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    
    return data, metadata


def fit_residual_model(input_path, output_path, k=5, visualize=True):
    """
    Fit KNN residual model from collected data.
    
    Args:
        input_path: Path to collected data pickle
        output_path: Where to save residual model
        k: Number of neighbors for KNN
        visualize: Plot residuals
    """
    print("="*70)
    print("FITTING DYNAMICS RESIDUAL MODEL")
    print("="*70)
    print(f"Input: {input_path}")
    print(f"K-neighbors: {k}")
    print("="*70)
    
    # Load data
    print("\n📂 Loading data...")
    with open(input_path, 'rb') as f:
        saved = pickle.load(f)
    
    data = saved['data']
    metadata = saved['metadata']
    
    print(f"✓ Loaded {len(data['gt_position'])} samples")
    print(f"  Duration: {metadata['duration']}s")
    print(f"  Collection date: {metadata['collection_date']}")
    
    # Compute ground truth accelerations
    print("\n🔬 Computing residuals...")
    
    dt = metadata['control_dt']  # 0.04s
    gt_velocity = data['gt_velocity']
    
    # Ground truth acceleration: a = (v[t+1] - v[t]) / dt
    a_true = np.diff(gt_velocity, axis=0) / dt  # (N-1, 3)
    
    # Predicted acceleration from simulator
    # For simplicity, assume perfect tracking: a_sim ≈ (cmd_vel - curr_vel) * gains
    # In reality, you'd run your simulator in parallel
    # Here's a simplified version:
    
    curr_vel = gt_velocity[:-1]  # (N-1, 3)
    cmd_vel = data['cmd_velocity'][:-1]  # (N-1, 3)
    
    # Simple proportional controller approximation
    # Real simulator would be more complex
    vel_error = cmd_vel - curr_vel
    gains = np.array([1.8, 1.8, 4.0])  # vel_p_xy, vel_p_xy, vel_p_z
    a_sim = vel_error * gains
    
    # Residual = true - predicted
    a_residual = a_true - a_sim
    
    print(f"✓ Computed {len(a_residual)} residual samples")
    print(f"  Mean residual: [{a_residual[:, 0].mean():.3f}, "
          f"{a_residual[:, 1].mean():.3f}, {a_residual[:, 2].mean():.3f}] m/s²")
    print(f"  Std residual:  [{a_residual[:, 0].std():.3f}, "
          f"{a_residual[:, 1].std():.3f}, {a_residual[:, 2].std():.3f}] m/s²")
    
    # Features: current velocity + commanded velocity
    features = np.hstack([curr_vel, cmd_vel])  # (N-1, 6)
    
    # Fit KNN
    print(f"\n🤖 Fitting KNN with k={k}...")
    knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
    knn.fit(features, a_residual)
    
    # Evaluate
    score = knn.score(features, a_residual)
    print(f"✓ Model fitted!")
    print(f"  R² score: {score:.4f}")
    print(f"  Training samples: {len(features)}")
    
    # Cross-validation
    from sklearn.model_selection import cross_val_score
    cv_scores = cross_val_score(knn, features, a_residual, cv=5, 
                                 scoring='r2')
    print(f"  CV R² scores: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Save model
    print(f"\n💾 Saving model to {output_path}...")
    
    residual_model = {
        'knn_model': knn,
        'n_samples': len(features),
        'k_neighbors': k,
        'residual_stats': {
            'mean': a_residual.mean(axis=0).tolist(),
            'std': a_residual.std(axis=0).tolist(),
            'max': np.abs(a_residual).max(axis=0).tolist()
        },
        'r2_score': float(score),
        'cv_scores': cv_scores.tolist(),
        'metadata': metadata,
        'fit_date': datetime.now().isoformat()
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(residual_model, f)
    
    print(f"✓ Model saved!")
    
    # Visualization
    if visualize:
        print("\n📊 Generating visualizations...")
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle('Dynamics Residuals', fontsize=16)
        
        time = np.arange(len(a_residual)) * dt
        
        labels = ['X (forward)', 'Y (right)', 'Z (up)']
        for i, ax in enumerate(axes):
            ax.plot(time, a_residual[:, i], 'b-', alpha=0.6, linewidth=0.5)
            ax.axhline(0, color='k', linestyle='--', alpha=0.3)
            ax.fill_between(time, 
                           a_residual[:, i].mean() - a_residual[:, i].std(),
                           a_residual[:, i].mean() + a_residual[:, i].std(),
                           alpha=0.2, color='blue')
            ax.set_ylabel(f'Residual {labels[i]} [m/s²]')
            ax.grid(True, alpha=0.3)
            
            # Stats text
            mean = a_residual[:, i].mean()
            std = a_residual[:, i].std()
            ax.text(0.02, 0.98, f'μ={mean:.3f}, σ={std:.3f}',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        axes[-1].set_xlabel('Time [s]')
        plt.tight_layout()
        
        plot_path = output_path.replace('.pkl', '_residuals.png')
        plt.savefig(plot_path, dpi=150)
        print(f"✓ Plot saved to {plot_path}")
        
        plt.show()
    
    print("\n" + "="*70)
    print("✅ RESIDUAL MODEL FITTING COMPLETE!")
    print("="*70)
    print(f"Ready for fine-tuning with: {output_path}")
    print("="*70)
    
    return residual_model


def main():
    parser = argparse.ArgumentParser(
        description='Collect and fit residual models for sim-to-real transfer'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Collect command
    collect_parser = subparsers.add_parser('collect', help='Collect data from Gazebo')
    collect_parser.add_argument('--policy', type=str, required=True,
                               help='Path to saved policy')
    collect_parser.add_argument('--output', type=str, default='gazebo_data.pkl',
                               help='Output pickle file')
    collect_parser.add_argument('--duration', type=int, default=60,
                               help='Flight duration in seconds')
    
    # Fit command
    fit_parser = subparsers.add_parser('fit', help='Fit residual model from data')
    fit_parser.add_argument('--input', type=str, required=True,
                           help='Input data pickle')
    fit_parser.add_argument('--output', type=str, default='dynamics_residual_model.pkl',
                           help='Output model pickle')
    fit_parser.add_argument('--k', type=int, default=5,
                           help='Number of neighbors for KNN')
    fit_parser.add_argument('--no-plot', action='store_true',
                           help='Skip visualization')
    
    args = parser.parse_args()
    
    if args.command == 'collect':
        collect_data(
            policy_path=args.policy,
            output_path=args.output,
            duration=args.duration
        )
    elif args.command == 'fit':
        fit_residual_model(
            input_path=args.input,
            output_path=args.output,
            k=args.k,
            visualize=not args.no_plot
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    import os
    main()
