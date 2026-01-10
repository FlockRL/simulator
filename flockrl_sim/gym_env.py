"""
Gymnasium environment wrapper around the CoreSimulator.

This environment makes it easy to plug the existing simulator into standard
RL tooling that expects the Gymnasium API.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import SimulationConfig
from .environment import Environment
from .simulator import CoreSimulator
from .state import SwarmState

DEFAULT_ACCELERATION_LIMIT = 5.0


class FlockRLGymEnv(gym.Env):
    """Single-drone Gymnasium environment backed by the CoreSimulator."""

    metadata = {"render_modes": ["none"], "render_fps": 60}

    def __init__(
        self,
        environment: Optional[Environment] = None,
        sim_config: Optional[SimulationConfig] = None,
        max_neighbors: int = 4,
        success_reward: float = 100.0,
        collision_penalty: float = 50.0,
        step_cost: float = 0.1,
        distance_scale: float = 1.0,
    ) -> None:
        """
        Args:
            environment: Optional environment instance. If not provided, a simple empty
                environment is created using the defaults from CoreSimulator.
            sim_config: Simulation configuration (dt, termination rules, etc.).
            max_neighbors: Maximum number of neighbors to encode in observations.
            success_reward: Bonus reward when reaching the goal.
            collision_penalty: Penalty applied when the episode ends due to collision.
            step_cost: Constant cost subtracted each step to encourage faster completion.
            distance_scale: Scale factor on dense reward based on goal distance reduction.
        """
        super().__init__()
        self.environment = environment or Environment(
            bounds=(-100, 100, -100, 100, 0, 100),
            obstacles=[],
            start_position=(0.0, 0.0, 1.0),
            goal_position=(0.0, 0.0, 10.0),
            seed=0,
        )
        self.sim_config = sim_config or SimulationConfig()
        self.simulator = CoreSimulator(
            delta_t=self.sim_config.delta_t,
            environment=self.environment,
            config=self.sim_config,
        )

        self.max_neighbors = max_neighbors
        self.success_reward = success_reward
        self.collision_penalty = collision_penalty
        self.step_cost = step_cost
        self.distance_scale = distance_scale

        self._action_limit = float(
            self.sim_config.max_acceleration or DEFAULT_ACCELERATION_LIMIT
        )
        self._num_rays = (
            self.simulator._perception_system.config.num_rays
            if self.simulator._perception_system is not None
            else 0
        )
        self._last_goal_distance: Optional[float] = None

        self.action_space = spaces.Box(
            low=-self._action_limit,
            high=self._action_limit,
            shape=(3,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._observation_dim(),),
            dtype=np.float32,
        )

    def _observation_dim(self) -> int:
        # pos(3) + vel(3) + goal vector(3) + goal distance(1)
        base = 10
        sensor = self._num_rays * 2  # ranges + hits
        neighbors = self.max_neighbors * 6  # relative position + velocity per neighbor
        return base + sensor + neighbors

    def _initial_state(self) -> SwarmState:
        pos = np.array([self.environment.start_position], dtype=float)
        ids = np.array([0], dtype=int)
        goals = np.array([self.environment.goal_position], dtype=float)
        return SwarmState.from_initial_positions(pos, ids, goals)

    def _goal_distance(self, state: SwarmState) -> float:
        return float(np.linalg.norm(state.pos[0] - state.goals[0]))

    def _build_observation(
        self, state: SwarmState, sim_info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        if sim_info is not None and sim_info.get("observations"):
            readings = sim_info["observations"]
        elif self.simulator._perception_system is not None:
            readings = self.simulator._perception_system.observe(state)
        else:
            readings = []

        reading = readings[0] if readings else None
        if reading:
            ranges = reading.ranges.astype(np.float32)
            hits = reading.hits.astype(np.float32)
            neighbor_vectors = reading.neighbor_vectors.astype(np.float32)
        else:
            ranges = np.zeros(self._num_rays, dtype=np.float32)
            hits = np.zeros(self._num_rays, dtype=np.float32)
            neighbor_vectors = np.zeros((0, 6), dtype=np.float32)

        neighbor_vectors = neighbor_vectors[: self.max_neighbors]
        if neighbor_vectors.shape[0] < self.max_neighbors:
            pad = np.zeros(
                (self.max_neighbors - neighbor_vectors.shape[0], 6), dtype=np.float32
            )
            neighbor_vectors = (
                np.vstack([neighbor_vectors, pad])
                if neighbor_vectors.size
                else pad.astype(np.float32)
            )

        goal_vector = (state.goals[0] - state.pos[0]).astype(np.float32)
        vel = (
            state.vel[0].astype(np.float32)
            if state.vel is not None
            else np.zeros(3, dtype=np.float32)
        )

        obs_parts = [
            state.pos[0].astype(np.float32),
            vel,
            goal_vector,
            np.array([self._goal_distance(state)], dtype=np.float32),
            ranges,
            hits,
            neighbor_vectors.flatten(),
        ]
        return np.concatenate(obs_parts, dtype=np.float32)

    def _compute_reward(self, state: SwarmState, sim_info: Dict[str, Any]) -> float:
        current_dist = self._goal_distance(state)
        delta = (
            self._last_goal_distance - current_dist
            if self._last_goal_distance is not None
            else 0.0
        )
        reward = delta * self.distance_scale - self.step_cost

        termination_reason = sim_info.get("termination_reason")
        if termination_reason == "success":
            reward += self.success_reward
        elif termination_reason == "collision":
            reward -= self.collision_penalty

        self._last_goal_distance = current_dist
        return float(reward)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        state = self.simulator.start_run(
            initial_state=self._initial_state(),
            metadata={"reset_seed": seed, "reset_options": options or {}},
        )
        self._last_goal_distance = self._goal_distance(state)
        obs = self._build_observation(state)
        info = {
            "goal_distance": self._last_goal_distance,
            "termination_reason": None,
            "collisions": [],
        }
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (3,):
            raise ValueError(f"Expected action shape (3,), got {action.shape}")

        clipped_action = np.clip(action, -self._action_limit, self._action_limit)
        state, sim_info = self.simulator.step(clipped_action[None, :])

        obs = self._build_observation(state, sim_info)
        reward = self._compute_reward(state, sim_info)

        terminated = bool(sim_info.get("done")) and (
            sim_info.get("termination_reason") != "timeout"
        )
        truncated = bool(sim_info.get("done")) and (
            sim_info.get("termination_reason") == "timeout"
        )

        info = {
            "goal_distance": sim_info["episode_stats"]["final_goal_distance"],
            "termination_reason": sim_info.get("termination_reason"),
            "collisions": sim_info.get("collisions", []),
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        # Offline rendering is handled by the simulator's logger/visualizer.
        return None

    def close(self) -> None:
        return None
