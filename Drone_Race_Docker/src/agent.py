# src/agent.py
"""PPO agent creation and configuration."""

import tensorflow as tf
from tf_agents.agents.ppo import ppo_agent
from tf_agents.networks import actor_distribution_network, value_network


def create_ppo_agent(tf_env, learning_rate=3e-4):
    """
    Create PPO agent with Swift paper architecture.
    
    Args:
        tf_env: TF-Agents environment
        learning_rate: Adam optimizer learning rate
    
    Returns:
        Initialized PPO agent
    """
    
    ACTOR_FC_LAYERS = (128, 128)
    VALUE_FC_LAYERS = (128, 128)
    
    # Actor network
    actor_net = actor_distribution_network.ActorDistributionNetwork(
        tf_env.observation_spec(),
        tf_env.action_spec(),
        fc_layer_params=ACTOR_FC_LAYERS,
        activation_fn=tf.nn.leaky_relu,
        kernel_initializer='glorot_uniform',
    )
    
    # Value network
    value_net = value_network.ValueNetwork(
        tf_env.observation_spec(),
        fc_layer_params=VALUE_FC_LAYERS,
        activation_fn=tf.nn.leaky_relu,
        kernel_initializer='glorot_uniform',
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    agent = ppo_agent.PPOAgent(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        optimizer=optimizer,
        actor_net=actor_net,
        value_net=value_net,
        num_epochs=10,
        discount_factor=0.99,
        lambda_value=0.95,
        importance_ratio_clipping=0.2,
        entropy_regularization=0.01,
        use_gae=True,
        use_td_lambda_return=True,
        normalize_observations=False,  # Keep OFF for your sim
        normalize_rewards=False,        # Keep OFF for your sim
        
        # STABILITY ADDITIONS:
        gradient_clipping=0.5,          # Emergency brake for gradients
        policy_l2_reg=0.0001,           # Keep policy weights reasonable
        value_function_l2_reg=0.0001,   # Keep value weights reasonable
        value_pred_loss_coef=0.25,      # Reduce from 0.5
        adaptive_kl_target=0.01,        # Limit policy changes
        adaptive_kl_tolerance=0.3,      # Early stopping threshold
        
        train_step_counter=tf.Variable(0, dtype=tf.int32),
    )
    
    agent.initialize()
    
    return agent, optimizer, actor_net, value_net