import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tf_agents.system import system_multiprocessing
from tf_agents.environments import parallel_py_environment, tf_py_environment

from src.tf_agents_wrapper import TFAgentsCopterEnv
from src.agent import create_ppo_agent
from src.trainer import PPOTrainer

# Enable multiprocessing
system_multiprocessing.enable_interactive_mode()

def main():
    """Main training function."""
    
    # Configuration
    NUM_PARALLEL_ENVS = 20
    TOTAL_TIMESTEPS = 500_000
    CHECKPOINT_FREQ = 10_000
    
    print("="*70)
    print("DRONE RACING RL TRAINING")
    print("="*70)
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPUs: {tf.config.list_physical_devices('GPU')}")
    print(f"Parallel Envs: {NUM_PARALLEL_ENVS}")
    print("="*70)
    
    # Create parallel environments
    def env_constructor():
        return TFAgentsCopterEnv(use_gimbal=True)
    
    parallel_env = parallel_py_environment.ParallelPyEnvironment(
        [env_constructor] * NUM_PARALLEL_ENVS
    )
    
    tf_env = tf_py_environment.TFPyEnvironment(parallel_env)
    
    print(f"\n✅ Created {NUM_PARALLEL_ENVS} parallel environments")
    
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
        checkpoint_freq=CHECKPOINT_FREQ
    )
    
    # Run training
    try:
        trainer.visualize_episode_tfagents(max_steps=1_500)
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted")
        current = trainer.global_step.numpy()
        trainer.train_checkpointer.save(global_step=current)
        print(f"✅ Checkpoint saved at step {current:,}")


if __name__ == "__main__":
    main()