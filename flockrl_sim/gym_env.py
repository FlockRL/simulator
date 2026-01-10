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

from .collision.system import CollisionSystem
from .config import SimulationConfig
from .environment import Environment, EnvironmentBuilder
from .environment.loader import EnvironmentSpecLoader
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
        drone_radius: float = 0.5,
        enable_collisions: bool = True,
    ) -> None:
        """
        Initialize the FlockRL Gymnasium environment.

        Args:
            environment: Optional environment instance. If not provided, loads the
                'simple' environment spec (2 walls with gates, 3 random clutter).
            sim_config: Simulation configuration (dt, termination rules, etc.).
            max_neighbors: Maximum number of neighbors to encode in observations.
            success_reward: Bonus reward when reaching the goal.
            collision_penalty: Penalty applied when the episode ends due to collision.
            step_cost: Constant cost subtracted each step to encourage faster completion.
            distance_scale: Scale factor on dense reward based on goal distance reduction.
            drone_radius: Radius of the drone for collision detection (meters).
            enable_collisions: Whether to enable collision detection and response.

        Note:
            The environment (including random obstacle placement) is built once during
            initialization using the environment spec's seed. The seed parameter in
            reset() only affects the drone's initial state (position/velocity), not
            obstacle layout. To get different obstacle configurations, create a new
            environment instance or provide a custom environment with a different seed.
        """
        super().__init__()

        # Load default environment from simple.json spec if not provided
        if environment is None:
            loader = EnvironmentSpecLoader()
            spec = loader.load("simple")
            environment = EnvironmentBuilder.from_spec(spec).build()

        self.environment = environment
        self.sim_config = sim_config or SimulationConfig()

        # Create collision system if enabled
        collision_system = None
        if enable_collisions:
            collision_system = CollisionSystem(
                environment=self.environment,
                drone_radius=drone_radius,
            )

        self.simulator = CoreSimulator(
            delta_t=self.sim_config.delta_t,
            environment=self.environment,
            config=self.sim_config,
            collision_system=collision_system,
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
        """
        Reset the environment to initial state.

        Args:
            seed: Random seed for reproducibility. Note that this only affects the
                drone's initial state, not the obstacle layout (which is fixed at
                environment initialization).
            options: Additional options (currently unused).

        Returns:
            Tuple of (observation, info dict).
        """
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
