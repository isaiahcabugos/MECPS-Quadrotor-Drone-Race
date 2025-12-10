# test_train.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tf_agents.system import system_multiprocessing
from tf_agents.environments import parallel_py_environment, tf_py_environment

from src.tf_agents_wrapper import TFAgentsCopterEnv
from src.agent import create_ppo_agent
from src.trainer import PPOTrainer

system_multiprocessing.enable_interactive_mode()

def test_training():
    print("Creating environments...")
    
    NUM_ENVS = 4  # Small number for testing
    
    def env_constructor():
        return TFAgentsCopterEnv(use_gimbal=True)
    
    parallel_env = parallel_py_environment.ParallelPyEnvironment(
        [env_constructor] * NUM_ENVS
    )
    
    tf_env = tf_py_environment.TFPyEnvironment(parallel_env)
    print(f"✅ Created {NUM_ENVS} environments")
    
    print("Creating agent...")
    agent, optimizer, _, _ = create_ppo_agent(tf_env)
    print("✅ Agent created")
    
    print("Creating trainer...")
    trainer = PPOTrainer(
        agent=agent,
        tf_env=tf_env,
        optimizer=optimizer,
        checkpoint_dir='/workspace/test_checkpoints',
        checkpoint_freq=10000
    )
    print("✅ Trainer created")
    
    print("\nRunning test training (60k steps)...")
    trainer.train(total_timesteps=60000, n_steps_per_env=1500)
    
    print("\n✅ TEST PASSED - Training completed successfully!")

if __name__ == "__main__":
    test_training()