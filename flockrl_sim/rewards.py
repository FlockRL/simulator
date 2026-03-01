"""
Reward function base class and built-in reward functions for FlockRL.
"""

from typing import Any, Dict

import numpy as np

from .state import SwarmState


class RewardFunction:
    """
    Base class for reward functions.

    Subclass this to define custom reward shaping for your training.
    Implement the `compute()` method (and optionally `reset()` if you need state tracking).
    """

    def reset(self, state: SwarmState) -> None:
        """Called when environment is reset. Override for state tracking."""
        pass

    def compute(
        self, state: SwarmState, action: np.ndarray, sim_info: Dict[str, Any]
    ) -> np.ndarray:
        """
        Compute rewards for all drones in the current step.

        Args:
            state: Current simulation state with N drones
            action: Actions that were taken (after clipping), shape (N, 3)
            sim_info: Dictionary with simulation information including:
                - termination_reason: How episode ended (if done)
                - episode_stats: Statistics about the episode
                - collisions: List of collision events

        Returns:
            Reward array of shape (N,) - one reward per drone
        """
        raise NotImplementedError("Subclasses must implement compute()")


class ProgressReward(RewardFunction):
    """
    Reward function with strong progress signal toward goal.

    Rewards getting closer to the goal, penalizes collisions,
    and gives a large bonus for reaching the goal.
    """

    def __init__(
        self,
        progress_scale: float = 20.0,
        step_penalty: float = 0.05,
        success_reward: float = 200.0,
        collision_penalty: float = -50.0,
        wall_collision_penalty: float = -50.0,
        obstacle_collision_penalty: float = -50.0,
        alive_bonus: float = 0.1,
        jerk_penalty: float = 0.0
    ):
        self.progress_scale = progress_scale
        self.step_penalty = step_penalty
        self.success_reward = success_reward
        self.collision_penalty = collision_penalty
        self.wall_collision_penalty = wall_collision_penalty
        self.obstacle_collision_penalty = obstacle_collision_penalty
        self.alive_bonus = alive_bonus
        self.jerk_penalty = jerk_penalty
        self._last_dist = None
        self._last_action = None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ProgressReward":
        """Create from the 'reward' section of config.yml."""
        reward_cfg = config.get("reward", {})
        return cls(
            progress_scale=reward_cfg.get("progress_scale", 20.0),
            step_penalty=reward_cfg.get("step_penalty", 0.05),
            success_reward=reward_cfg.get("success_reward", 200.0),
            collision_penalty=reward_cfg.get("collision_penalty", -50.0),
            wall_collision_penalty=reward_cfg.get("wall_collision_penalty", -50.0),
            obstacle_collision_penalty=reward_cfg.get("obstacle_collision_penalty", -50.0),
            alive_bonus=reward_cfg.get("alive_bonus", 0.1),
            jerk_penalty=reward_cfg.get("jerk_penalty", 0.0),
        )

    def reset(self, state: SwarmState) -> None:
        self._last_dist = np.linalg.norm(state.pos - state.goals, axis=1)
        self._last_action = np.zeros((state.pos.shape[0], 3), dtype=np.float32)

    def compute(self, state: SwarmState, action: np.ndarray, sim_info: Dict[str, Any]) -> np.ndarray:
        curr_dist = np.linalg.norm(state.pos - state.goals, axis=1)
        progress = self._last_dist - curr_dist
        rewards = self.progress_scale * progress - self.step_penalty + self.alive_bonus

        # Jerk Penalty Calculations:
        action_diff = np.linalg.norm(action - self._last_action, axis=1)
        rewards -= self.jerk_penalty * action_diff
        
        # Save the current action for the next step's calculation
        self._last_action = action.copy()

        if sim_info["termination_reason"] == "success":
            rewards += self.success_reward
        elif sim_info["termination_reason"] == "collision":
            collisions = sim_info.get("collisions", [])
            wall = any(c.collision_type == "wall" for c in collisions)
            obstacle = any(c.collision_type in ("clutter", "sphere") for c in collisions)

            if wall:
                rewards += self.wall_collision_penalty
            elif obstacle:
                rewards += self.obstacle_collision_penalty
            else:
                rewards += self.collision_penalty

        self._last_dist = curr_dist
        return rewards


__all__ = ["RewardFunction", "ProgressReward"]
