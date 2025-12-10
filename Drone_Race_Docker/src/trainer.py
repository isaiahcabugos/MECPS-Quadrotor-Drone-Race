# src/trainer.py
"""Training loop for PPO agent with full feature parity."""

import os
import time
import json
from datetime import datetime
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.utils import common
from tf_agents.policies import policy_saver

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.tf_agents_wrapper import TFAgentsCopterEnv

try:
    import psutil
    import GPUtil
    HAS_MONITORING = True
except ImportError:
    HAS_MONITORING = False


class PPOTrainer:
    """Handles PPO training loop with full feature parity to notebook."""
    
    def __init__(self, agent, tf_env, optimizer, 
                 checkpoint_dir='./checkpoints',
                 log_dir='./logs',
                 checkpoint_freq=1_000_000):
        
        self.agent = agent
        self.tf_env = tf_env
        self.optimizer = optimizer
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.checkpoint_freq = checkpoint_freq
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        self.global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False)
        
        # Will be set in train()
        self.replay_buffer = None
        self.train_checkpointer = None
        self.tf_policy_saver = policy_saver.PolicySaver(agent.policy)
        
    def _setup_checkpointing(self):
        """Setup checkpointer (called after replay_buffer is created)."""
        self.train_checkpointer = common.Checkpointer(
            ckpt_dir=self.checkpoint_dir,
            max_to_keep=5,
            agent=self.agent,
            policy=self.agent.policy,
            global_step=self.global_step,
            replay_buffer=self.replay_buffer,  # RESTORED
            optimizer=self.optimizer
        )
    
    def _monitor_memory(self):
        """Monitor memory usage if psutil/GPUtil available."""
        if not HAS_MONITORING:
            return
        
        try:
            ram = psutil.virtual_memory()
            process = psutil.Process(os.getpid())
            process_ram_gb = process.memory_info().rss / (1024**3)
            
            print(f"\n  💾 Memory: {process_ram_gb:.2f} GB ({ram.percent:.1f}% system)")
            
            try:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    print(f"     GPU {gpu.id}: {gpu.memoryUsed:.0f}/{gpu.memoryTotal:.0f} MB ({gpu.memoryUtil*100:.1f}%)")
            except:
                pass
        except:
            pass

    def _get_avg_return(self, episode_returns, window=100):
        """Calculate average return with proper handling of edge cases."""
        if len(episode_returns) == 0:
            return 0.0
        elif len(episode_returns) < window:
            return np.mean(episode_returns)
        else:
            return np.mean(episode_returns[-window:])
    
    def train(self, total_timesteps=20_000_000, n_steps_per_env=1500):
        """Run training loop with full feature parity."""
        
        num_parallel_envs = self.tf_env.batch_size
        collect_steps_per_iteration = n_steps_per_env * num_parallel_envs
        num_iterations = int(total_timesteps / collect_steps_per_iteration)
        
        # Setup replay buffer
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            self.agent.collect_data_spec,
            batch_size=self.tf_env.batch_size,
            max_length=n_steps_per_env,
        )
        
        # Setup checkpointing AFTER replay buffer exists
        self._setup_checkpointing()
        
        # Setup collect driver
        collect_driver = dynamic_step_driver.DynamicStepDriver(
            self.tf_env,
            self.agent.collect_policy,
            observers=[self.replay_buffer.add_batch],
            num_steps=collect_steps_per_iteration,
        )
        
        # Load checkpoint if exists
        self.train_checkpointer.initialize_or_restore()
        loaded_step = self.global_step.numpy()
        self.agent.train_step_counter.assign(loaded_step)
        current_step = loaded_step
        
        start_iteration = current_step // collect_steps_per_iteration
        
        # Tracking metrics (RESTORED from original)
        episode_returns = []
        total_episodes = 0  # RESTORED
        total_crashes = 0   # RESTORED
        total_gates_passed = 0  # RESTORED
        crash_reasons = {}  # RESTORED
        last_checkpoint_step = current_step
        
        print(f"\n{'='*70}")
        print(f"STARTING TRAINING")
        print(f"{'='*70}")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Parallel envs: {num_parallel_envs}")
        print(f"Starting from step: {current_step:,}")
        print(f"Remaining: {total_timesteps - current_step:,}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        
        try:
            for iteration in tqdm(range(start_iteration, num_iterations),
                                desc="Training",
                                initial=start_iteration,
                                total=num_iterations):
                
                if current_step >= total_timesteps:
                    break
                
                iteration_start = time.time()
                
                # Collect trajectories
                collect_start = time.time()
                self.replay_buffer.clear()
                collect_driver.run()
                collect_time = time.time() - collect_start
                
                # Train
                trajectories = self.replay_buffer.gather_all()
                train_start = time.time()
                train_loss = self.agent.train(experience=trajectories)
                train_time = time.time() - train_start
                
                # Update counters
                current_step += collect_steps_per_iteration
                self.global_step.assign(current_step)
                self.agent.train_step_counter.assign(current_step)
                
                # Extract metrics (RESTORED)
                traj_rewards = trajectories.reward.numpy()
                episode_reward_estimate = np.mean(np.sum(traj_rewards, axis=1))
                episode_returns.append(episode_reward_estimate)
                
                terminal_mask = trajectories.is_last().numpy()
                episodes_completed = int(np.sum(terminal_mask))
                total_episodes += episodes_completed
                
                iteration_time = time.time() - iteration_start
                loss_value = train_loss.loss.numpy()
                
                # Logging
                if iteration % 10 == 0:
                    avg_return = self._get_avg_return(episode_returns)  # Use helper
                    elapsed = time.time() - start_time
                    fps = current_step / elapsed if elapsed > 0 else 0
                    
                    print(f"\n{'='*70}")
                    print(f"Iteration {iteration+1}/{num_iterations} (Step {current_step:,}/{total_timesteps:,})")
                    print(f"{'='*70}")
                    print(f"Progress: {100*current_step/total_timesteps:.1f}%")
                    print(f"Loss: {loss_value:.4f}")
                    print(f"Avg Return (100): {avg_return:.3f}")
                    print(f"Episodes: {episodes_completed} (Total: {total_episodes})")
                    print(f"Collect: {collect_time:.2f}s ({collect_steps_per_iteration/collect_time:.0f} FPS)")
                    print(f"Train: {train_time:.2f}s")
                    print(f"Overall FPS: {fps:.0f}")
                    self._monitor_memory()
                    if fps > 0:
                        remaining_min = (total_timesteps - current_step) / fps / 60
                        print(f"Est. Remaining: {remaining_min:.1f} min ({remaining_min/60:.1f} hrs)")
                    print(f"{'='*70}")
                else:
                    avg_return = self._get_avg_return(episode_returns)  # Use helper
                    print(f"Iter {iteration+1}: Loss={loss_value:.4f}, Return={avg_return:.2f}, Episodes={episodes_completed}")

                # Checkpointing
                if current_step - last_checkpoint_step >= self.checkpoint_freq:
                    self._save_checkpoint(current_step, episode_returns, total_episodes)
                    last_checkpoint_step = current_step
            
            # Final save
            self._save_checkpoint(current_step, episode_returns, total_episodes, final=True)
            
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted - saving checkpoint...")
            self._save_checkpoint(current_step, episode_returns, total_episodes)
            raise
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            print("Saving emergency checkpoint...")
            try:
                self._save_checkpoint(current_step, episode_returns, total_episodes)
            except:
                print("Failed to save emergency checkpoint")
            raise
        
        total_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"✅ TRAINING COMPLETE!")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Final step: {current_step:,}")
        print(f"Total episodes: {total_episodes}")
        print(f"Avg FPS: {current_step/total_time:.0f}")
        print(f"{'='*70}\n")
    
    def _save_checkpoint(self, current_step, episode_returns, total_episodes, final=False):
        """Save checkpoint and metrics."""
        
        print(f"\n💾 Saving checkpoint at step {current_step:,}...")
        
        self.train_checkpointer.save(global_step=current_step)
        
        policy_path = os.path.join(self.checkpoint_dir, 'policy', 
                                   f'policy_{"FINAL" if final else f"step_{current_step}"}')
        self.tf_policy_saver.save(policy_path)
        
        avg_return = np.mean(episode_returns[-100:]) if len(episode_returns) >= 100 else 0.0
        
        metrics = {
            'timesteps': int(current_step),
            'total_episodes': int(total_episodes),
            'avg_return_100': float(avg_return),
            'timestamp': datetime.now().isoformat(),
        }
        
        metrics_path = os.path.join(self.checkpoint_dir, f'metrics_step_{current_step}.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"✅ Checkpoint saved | Avg Return: {avg_return:.3f} | Episodes: {total_episodes}")
    
    def visualize_episode_tfagents(self, max_steps=500, total_timesteps=1_000_000, n_steps_per_env=1500):
        """
        See what the trained policy is actually doing.

        Args:
            agent: Trained PPO agent
            max_steps: Maximum steps to run (default 500, or full episode 1500)
        """
        num_parallel_envs = self.tf_env.batch_size
        collect_steps_per_iteration = n_steps_per_env * num_parallel_envs
        num_iterations = int(total_timesteps / collect_steps_per_iteration)
        
        # Setup replay buffer
        self.replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            self.agent.collect_data_spec,
            batch_size=self.tf_env.batch_size,
            max_length=n_steps_per_env,
        )
        

        # Setup checkpointing AFTER replay buffer exists
        self._setup_checkpointing()
        self.train_checkpointer.initialize_or_restore()
        # Create single evaluation environment
        env = TFAgentsCopterEnv(episode_length=1500)

        # Reset environment
        time_step = env.reset()

        # Access underlying simulator for detailed info
        sim = env._sim

        print("="*70)
        print("EPISODE VISUALIZATION")
        print("="*70)
        print(f"Start position: {sim.position}")
        print(f"Start velocity: {sim.velocity}")
        print(f"Target gate #{sim.current_gate_idx}: {sim.gates[sim.current_gate_idx]}")
        print(f"Initial distance: {np.linalg.norm(sim.gates[sim.current_gate_idx] - sim.position):.2f}m")
        print("="*70 + "\n")

        # Tracking arrays
        positions = [sim.position.copy()]
        velocities = [sim.velocity.copy()]
        actions_taken = []
        rewards = []
        gates_passed = []
        distances_to_gate = []

        for step in range(max_steps):
            # Get action from policy (deterministic)
            action_step = self.agent.policy.action(time_step)

            # Extract action (handle both tensor and numpy)
            if hasattr(action_step.action, 'numpy'):
                action = action_step.action.numpy().flatten()
            else:
                action = np.array(action_step.action).flatten()

            # Step environment
            time_step = env.step(action)

            # Handle reward (could be tensor or numpy array depending on environment)
            if hasattr(time_step.reward, 'numpy'):
                reward = float(time_step.reward.numpy())
            else:
                reward = float(time_step.reward)

            # Record data
            positions.append(sim.position.copy())
            velocities.append(sim.velocity.copy())
            actions_taken.append(action)
            rewards.append(reward)
            gates_passed.append(sim.gates_passed)

            dist = np.linalg.norm(sim.gates[sim.current_gate_idx] - sim.position)
            distances_to_gate.append(dist)

            # print(f"  r_progress={sim._last_r_progress:.3f}, r_perc={sim._last_r_perc:.3f}, "
            #     f"r_smooth={sim._last_r_smooth:.3f}, r_crash={sim._last_r_crash:.3f}")

            # Periodic logging
            if step < 10 or step % 500 == 0:
                print(f"Step {step:3d}: pos=[{sim.position[0]:6.2f}, {sim.position[1]:6.2f}, {sim.position[2]:6.2f}]  "
                    f"vel=[{sim.velocity[0]:5.2f}, {sim.velocity[1]:5.2f}, {sim.velocity[2]:5.2f}]")
                print(f"          action={action}  reward={reward:7.3f}")
                print(f"          gate={sim.current_gate_idx}  dist={dist:.2f}m  "
                    f"gates_passed={sim.gates_passed}")

                if step > 0 and gates_passed[-1] > gates_passed[-2]:
                    print(f"          ✅ PASSED GATE #{sim.current_gate_idx-1}!")
                print()


            # Check
            if time_step.is_last():
                # Determine termination reason from simulator state
                if sim.position[2] < 0.05:
                    reason = "ground_crash"
                elif np.any(np.abs(sim.position[:2]) > 15):
                    reason = "out_of_bounds"
                elif sim.position[2] > 6:
                    reason = "ceiling"
                elif sim.orientation[2, 2] < 0.5:
                    reason = "flipped"
                elif step >= max_steps:
                    reason = "max_steps_reached"
                else:
                    reason = "timeout"

                print("="*70)
                print(f"EPISODE ENDED at step {step}: {reason}")
                print("="*70)
                break

        # ========================================
        # Summary Statistics
        # ========================================
        positions = np.array(positions)
        velocities = np.array(velocities)
        actions_taken = np.array(actions_taken)

        total_displacement = np.linalg.norm(positions[-1] - positions[0])
        max_speed = np.max(np.linalg.norm(velocities, axis=1))
        avg_speed = np.mean(np.linalg.norm(velocities, axis=1))

        print(f"\n{'='*70}")
        print("EPISODE SUMMARY")
        print("="*70)
        print(f"Total steps: {step+1}")
        print(f"Gates passed: {sim.gates_passed} / {sim.num_gates}")
        print(f"Total reward: {sum(rewards):.2f}")
        print(f"Avg reward per step: {np.mean(rewards):.3f}")
        print(f"\n--- MOVEMENT ANALYSIS ---")
        print(f"Total displacement: {total_displacement:.2f}m")
        print(f"Max speed: {max_speed:.2f} m/s")
        print(f"Avg speed: {avg_speed:.2f} m/s")
        print(f"Avg action magnitude: {np.mean(np.linalg.norm(actions_taken, axis=1)):.3f}")
        print(f"Min distance to gate: {min(distances_to_gate):.2f}m")
        print("="*70)

        # ========================================
        # Diagnostic Checks
        # ========================================
        print(f"\n{'='*70}")
        print("DIAGNOSTIC CHECKS")
        print("="*70)

        # Check 1: Is drone moving?
        if total_displacement < 1.0:
            print("⚠️  WARNING: Drone barely moved! Policy may not be learning control.")
        else:
            print(f"✅ Drone is moving (displaced {total_displacement:.2f}m)")

        # Check 2: Is policy producing actions?
        avg_action_mag = np.mean(np.linalg.norm(actions_taken, axis=1))
        if avg_action_mag < 0.1:
            print("⚠️  WARNING: Actions are very small! Policy may be stuck.")
        else:
            print(f"✅ Policy producing actions (avg magnitude: {avg_action_mag:.3f})")

        # Check 3: Is drone reaching gates?
        if sim.gates_passed == 0 and min(distances_to_gate) > 5.0:
            print("⚠️  WARNING: Never got close to any gate! Policy not learning navigation.")
        elif sim.gates_passed == 0 and min(distances_to_gate) < 2.0:
            print(f"⚠️  Got close ({min(distances_to_gate):.2f}m) but didn't pass gate. Needs more training.")
        elif sim.gates_passed > 0:
            print(f"✅ Passed {sim.gates_passed} gate(s)! Policy is learning!")

        # Check 4: Reward trend
        if np.mean(rewards) < -1.0:
            print("⚠️  WARNING: Very negative rewards. Policy crashing frequently.")
        elif np.mean(rewards) < 0:
            print("⚠️  Negative avg reward. Policy needs more training.")
        else:
            print(f"✅ Positive avg reward ({np.mean(rewards):.2f})! Policy performing well.")

        print("="*70)

        # ========================================
        # Visualization (Optional)
        # ========================================
        plot_trajectory = input("\nPlot 3D trajectory? (y/n): ").lower().strip() == 'y'

        if plot_trajectory:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')

            # Plot trajectory
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                    'b-', linewidth=2, label='Drone trajectory')

            # Plot gates
            for i, gate_pos in enumerate(sim.gates):
                ax.scatter(*gate_pos, c='red', s=100, marker='o',
                        label='Gates' if i == 0 else '')
                ax.text(gate_pos[0], gate_pos[1], gate_pos[2] + 0.3,
                    f'G{i}', fontsize=10)

            # Mark start and end
            ax.scatter(*positions[0], c='green', s=200, marker='^',
                    label='Start', zorder=5)
            ax.scatter(*positions[-1], c='orange', s=200, marker='v',
                    label='End', zorder=5)

            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'Drone Trajectory (Gates passed: {sim.gates_passed}/{sim.num_gates})')
            ax.legend()
            ax.set_box_aspect([1, 1, 0.5])

            plt.tight_layout()
            plt.savefig('trajectory_debug.png', dpi=150)
            print("\n✅ Trajectory saved to 'trajectory_debug.png'")
            plt.show()