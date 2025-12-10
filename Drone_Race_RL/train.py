# train.py
"""Main training script for drone racing RL."""

import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tf_agents.system import system_multiprocessing
from tf_agents.environments import parallel_py_environment, tf_py_environment

from src.tf_agents_wrapper import TFAgentsCopterEnv
from src.agent import create_ppo_agent
from src.trainer import PPOTrainer

# Enable multiprocessing
system_multiprocessing.enable_interactive_mode()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train drone racing RL agent')
    
    parser.add_argument(
        '--total-steps',
        type=int,
        default=1_000_000,
        help='Total training timesteps (default: 1,000,000)'
    )
    
    parser.add_argument(
        '--checkpoint-freq',
        type=int,
        default=500_000,
        help='Save checkpoint every N steps (default: 500,000)'
    )
    
    parser.add_argument(
        '--num-envs',
        type=int,
        default=20,
        help='Number of parallel environments (default: 20)'
    )
    
    parser.add_argument(
        '--use-gimbal',
        action='store_true',
        default=True,
        help='Use gimbal-stabilized camera (default: True)'
    )
    
    return parser.parse_args()


def main():
    """Main training function."""
    
    args = parse_args()
    
    print("="*70)
    print("DRONE RACING RL TRAINING")
    print("="*70)
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPUs: {tf.config.list_physical_devices('GPU')}")
    print(f"\nConfiguration:")
    print(f"  Total Steps: {args.total_steps:,}")
    print(f"  Checkpoint Frequency: {args.checkpoint_freq:,}")
    print(f"  Parallel Envs: {args.num_envs}")
    print(f"  Use Gimbal: {args.use_gimbal}")
    print("="*70)
    
    # Create parallel environments
    def env_constructor():
        return TFAgentsCopterEnv(use_gimbal=args.use_gimbal)
    
    parallel_env = parallel_py_environment.ParallelPyEnvironment(
        [env_constructor] * args.num_envs
    )
    
    tf_env = tf_py_environment.TFPyEnvironment(parallel_env)
    
    print(f"\n✅ Created {args.num_envs} parallel environments")
    
    # Create agent
    agent, optimizer, actor_net, value_net = create_ppo_agent(tf_env)
    
    print("✅ PPO agent created")
    
    # Create trainer
    trainer = PPOTrainer(
        agent=agent,
        tf_env=tf_env,
        optimizer=optimizer,
        checkpoint_dir='./checkpoints',
        log_dir='./logs',
        checkpoint_freq=args.checkpoint_freq
    )
    
    # Run training
    try:
        trainer.train(total_timesteps=args.total_steps)
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted")
        current = trainer.global_step.numpy()
        trainer.train_checkpointer.save(global_step=current)
        print(f"✅ Checkpoint saved at step {current:,}")


if __name__ == "__main__":
    main()