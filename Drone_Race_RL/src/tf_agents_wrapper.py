# src/tf_agents_wrapper.py
"""TF-Agents environment wrapper for the Clover simulator."""

import numpy as np
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts

from src.simulator import ImprovedCopterSimulator, CloverSpecs, ControllerGains


class TFAgentsCopterEnv(py_environment.PyEnvironment):
    """
    TF-Agents wrapper around ImprovedCopterSimulator.
    """

    def __init__(self, seed=None, episode_length=1500, use_gimbal=True):
        super().__init__()
        
        self.episode_length = episode_length
        
        # FIXED: Import classes instead of redefining
        self._sim = ImprovedCopterSimulator(
            specs=CloverSpecs(),
            gains=ControllerGains(),
            dt=0.004,
            domain_randomization=True,
            add_noise=False,
            use_gimbal=use_gimbal
        )
        
        if seed is not None:
            self._sim.reset(seed=seed)
        
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

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        obs, _ = self._sim.reset()
        self._episode_ended = False
        self._episode_step = 0
        obs = np.array(obs, dtype=np.float32)
        assert obs.shape == (30,), f"Observation shape mismatch: {obs.shape}"
        return ts.restart(obs)

    def _step(self, action):
        if self._episode_ended:
            return self.reset()
        
        action = np.array(action, dtype=np.float32)
        
        if not np.all(np.abs(action) <= 1.0):
            action = np.clip(action, -1.0, 1.0)
        
        obs, reward, terminated, truncated, info = self._sim.step(action)
        self._episode_step += 1
        
        # Episode length enforcement
        if self._episode_step >= self.episode_length:
            truncated = True
            info['termination_reason'] = 'timeout_1500'
        
        obs = np.array(obs, dtype=np.float32)
        
        if terminated or truncated:
            self._episode_ended = True
            return ts.termination(obs, reward)
        else:
            return ts.transition(obs, reward, discount=0.99)

    def get_info(self):
        """Access simulator info for debugging/logging."""
        return {
            'episode_step': self._episode_step,
            'current_gate': self._sim.current_gate_idx,
            'gates_passed': self._sim.gates_passed,
            'position': self._sim.position.copy(),
            'velocity': self._sim.velocity.copy(),
            'orientation_euler': self._sim._rotation_to_euler(self._sim.orientation),
        }