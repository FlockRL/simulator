"""Gymnasium wrappers for FlockRL environments."""

from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .gym_env import FlockRLGymEnv


class SingleDroneWrapper(gym.Wrapper):
    """
    Wrapper to extract single-drone observations and actions from multi-drone environment.

    PPO/SB3 works with 1D arrays, but FlockRLGymEnv uses 2D arrays (num_drones dimension).
    This wrapper bridges the gap for single-drone training.
    """

    def __init__(self, env: FlockRLGymEnv):
        super().__init__(env)
        if env.num_drones != 1:
            raise ValueError(
                f"SingleDroneWrapper requires num_drones=1, got {env.num_drones}."
            )
        self.num_drones = env.num_drones
        self.obs_dim = env.observation_space.shape[1]
        self.action_dim = env.action_space.shape[1]

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=env.action_space.low[0],
            high=env.action_space.high[0],
            shape=(self.action_dim,),
            dtype=np.float32,
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        return obs[0], info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action_reshaped = action.reshape(1, self.action_dim)
        obs, rewards, terminated, truncated, info = self.env.step(action_reshaped)
        return obs[0], float(rewards[0]), terminated, truncated, info


__all__ = ["SingleDroneWrapper"]
