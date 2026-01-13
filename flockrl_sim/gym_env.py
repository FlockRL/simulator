"""
Gymnasium environment wrapper around the CoreSimulator.

This environment makes it easy to plug the existing simulator into standard
RL tooling that expects the Gymnasium API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from .collision.system import CollisionSystem
from .environment import Environment
from .environment.loader import EnvironmentSpecLoader
from .environment.obstacles import EnvironmentBuilder
from .gym_logging import EpisodeLogger
from .rewards import RewardFunction
from .simulator import CoreSimulator
from .state import SwarmState


def load_environment_from_spec(spec_name_or_path: Union[str, Path], config: Dict[str, Any]) -> Environment:
    """Load an environment from a preset name or JSON file path.
    
    This is a convenience function that combines EnvironmentSpecLoader and
    EnvironmentBuilder to create an Environment in one call.
    
    Args:
        spec_name_or_path: Preset name (e.g., "simple") or path to JSON spec file.
        config: Configuration dictionary with 'environment' section containing 
                'spawn_clearance' and 'max_placement_attempts'.
    
    Returns:
        Environment instance ready to use with FlockRLGymEnv.
    """
    loader = EnvironmentSpecLoader()
    spec = loader.load(spec_name_or_path)
    
    spawn_clearance = config["environment"]["spawn_clearance"]
    max_placement_attempts = config["environment"]["max_placement_attempts"]
    
    return EnvironmentBuilder.from_spec(
        spec, 
        spawn_clearance=spawn_clearance,
        max_placement_attempts=max_placement_attempts
    ).build()


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        # Look for config.yml in the project root (parent of flockrl_sim)
        config_path = Path(__file__).parent.parent / "config.yml"
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


class FlockRLGymEnv(gym.Env):
    """Multi-drone Gymnasium environment backed by the CoreSimulator."""

    def __init__(
        self,
        reward_fn: RewardFunction,
        environment: Environment,
        config_path: Optional[Path] = None,
    ) -> None:
        """
        Args:
            reward_fn: Reward function instance.
            environment: Environment instance.
            config_path: Optional path to config.yml file. If None, uses default location.
        """
        super().__init__()
        
        config = load_config(config_path)
        gym_config = config["gym"]
        sim_config = config["simulation"]
        collision_config = config["collision"]
        perception_config = config["perception"]
        visualization_config = config["visualization"]
        
        # Set metadata from config
        self.metadata = {
            "render_modes": ["none"],
            "render_fps": visualization_config["fps"]
        }
        
        self.environment = environment
        self.sim_config = sim_config
        self.perception_config = perception_config
        self.config = config
        env_config = config["environment"]
        self._max_spawn_attempts = int(env_config["max_placement_attempts"])
        
        # Create collision system
        collision_system = self._build_collision_system(collision_config)
        self._collision_system = collision_system
        
        log_dir = gym_config["log_dir"]
        self._save_runs = bool(gym_config["save_runs"])
        if self._save_runs and not log_dir:
            raise ValueError("gym.save_runs requires gym.log_dir to be set.")

        self.simulator = CoreSimulator(
            delta_t=sim_config["delta_t"],
            max_steps=sim_config["max_steps"],
            goal_threshold=sim_config["goal_threshold"],
            max_acceleration=sim_config["max_acceleration"],
            terminate_on_collision=sim_config["terminate_on_collision"],
            collision_system=collision_system,
            environment=self.environment,
            enable_frame_logging=self._save_runs,  # Only log frames if we'll save them
            perception_config=perception_config,
            reset_config=sim_config,
        )

        # Validate required gym config fields
        required_gym_fields = ["num_drones", "spawn_offset_range", "max_neighbors"]
        missing_fields = [field for field in required_gym_fields if field not in gym_config]
        if missing_fields:
            raise ValueError(
                f"Missing required gym config fields: {missing_fields}. "
                f"Please add them to your config.yml under 'gym' section."
            )

        self.num_drones = gym_config["num_drones"]
        self.spawn_offset_range = gym_config["spawn_offset_range"]
        self.max_neighbors = gym_config["max_neighbors"]
        self.reward_fn = reward_fn
        self._action_limit = float(sim_config["max_acceleration"])
        self._num_rays = (
            self.simulator._perception_system.config.num_rays
            if self.simulator._perception_system is not None
            else 0
        )

        # Episode logger (only enabled if log_dir is provided)
        self.logger: Optional[EpisodeLogger] = None
        if log_dir:
            self.logger = EpisodeLogger(log_dir=Path(log_dir))

        self._episode_num = 0
        self._episode_reward = np.zeros(1, dtype=np.float32)  # Will be resized in reset()

        self.action_space = spaces.Box(
            low=-self._action_limit,
            high=self._action_limit,
            shape=(self.num_drones, 3),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_drones, self._observation_dim()),
            dtype=np.float32,
        )

    def _build_collision_system(
        self, collision_config: Dict[str, Any]
    ) -> CollisionSystem:
        """Create a CollisionSystem instance (collisions are always enabled)."""
        restitution = collision_config["restitution"]
        drone_radius = collision_config["drone_radius"]
        return CollisionSystem(
            environment=self.environment,
            restitution=restitution,
            drone_radius=drone_radius
        )

    def _observation_dim(self) -> int:
        # vel(3) + goal vector(3) + goal distance(1)
        base = 7
        sensor = self._num_rays * 2  # ranges + hits
        neighbors = self.max_neighbors * 6  # relative position + velocity per neighbor
        return base + sensor + neighbors
    
    def _reset_perception(self, seed: int) -> None:
        if self.simulator._perception_system is None:
            return
        self.simulator._perception_system.reset(
            config=self.simulator._perception_system.config,
            seed=seed,
        )

    def _is_spawn_state_valid(self, state: SwarmState) -> bool:
        _, info = self._collision_system(state)
        return not info.get("collisions")

    def _initial_state(self) -> SwarmState:
        base_start = np.array(self.environment.start_position, dtype=float)
        base_goal = np.array(self.environment.goal_position, dtype=float)
        ids = np.arange(self.num_drones, dtype=int)
        position_noise = float(self.sim_config.get("reset_position_noise", 0.0) or 0.0)
        velocity_noise = float(self.sim_config.get("reset_velocity_noise", 0.0) or 0.0)
        attempts = max(1, self._max_spawn_attempts)
        for _ in range(attempts):
            # For multi-drone scenarios, add random offsets to prevent collisions at spawn.
            # For single drone, start exactly at the specified position for deterministic behavior.
            if self.num_drones > 1 and self.spawn_offset_range > 0:
                offsets = self._rng.uniform(
                    -self.spawn_offset_range,
                    self.spawn_offset_range,
                    size=(self.num_drones, 3),
                )
            else:
                offsets = np.zeros((self.num_drones, 3), dtype=float)

            pos = base_start[None, :] + offsets
            goals = base_goal[None, :] + offsets  # Same offset for goals to maintain relative positioning

            state = SwarmState.from_initial_positions(pos, ids, goals)
            if position_noise > 0.0:
                state.pos = state.pos + self._rng.normal(
                    0.0, position_noise, size=state.pos.shape
                )
            if velocity_noise > 0.0:
                state.vel = state.vel + self._rng.normal(
                    0.0, velocity_noise, size=state.vel.shape
                )

            if self._is_spawn_state_valid(state):
                return state

        raise ValueError(
            f"Unable to sample a valid spawn state after {attempts} attempts. "
            "Check spawn_offset_range, reset noise, or environment layout."
        )

    def _build_observation(
        self, state: SwarmState, sim_info: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        if sim_info is not None:
            readings = sim_info["observations"]
        elif self.simulator._perception_system is not None:
            readings = self.simulator._perception_system.observe(state)
        else:
            readings = []

        # Build observations for all drones
        observations = []
        for i in range(self.num_drones):
            reading = readings[i] if i < len(readings) else None
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

            goal_vector = (state.goals[i] - state.pos[i]).astype(np.float32)
            goal_distance = float(np.linalg.norm(goal_vector))
            vel = (
                state.vel[i].astype(np.float32)
                if state.vel is not None
                else np.zeros(3, dtype=np.float32)
            )

            obs_parts = [
                vel,
                goal_vector,
                np.array([goal_distance], dtype=np.float32),
                ranges,
                hits,
                neighbor_vectors.flatten(),
            ]
            observations.append(np.concatenate(obs_parts, dtype=np.float32))
        
        return np.array(observations, dtype=np.float32)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._rng = self.np_random
        if seed is not None:
            self._reset_perception(seed)

        state = self.simulator.start_run(
            initial_state=self._initial_state(),
            metadata={"reset_seed": seed, "reset_options": options or {}},
        )
        
        # Initialize reward function
        self.reward_fn.reset(state)
        
        obs = self._build_observation(state)
        goal_distances = np.linalg.norm(state.pos - state.goals, axis=1)
        info = {
            "goal_distance": goal_distances,
            "termination_reason": None,
            "collisions": [],
        }

        # Start episode logging
        if self.logger:
            metadata = {"seed": seed, **(options or {})}
            self.logger.start_episode(self._episode_num, metadata)

        self._episode_reward = 0.0

        return obs, info

    def reset_simulator(
        self, *, randomize: bool = False, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset using CoreSimulator.reset to reuse the stored initial state.
        Call reset() at least once before this.
        """
        if self.simulator.current_run is None:
            raise RuntimeError("Call reset() before reset_simulator().")

        super().reset(seed=seed)
        self._rng = self.np_random
        if seed is not None:
            self._reset_perception(seed)

        state = self.simulator.reset(randomize=randomize, seed=seed)
        self.reward_fn.reset(state)

        obs = self._build_observation(state)
        goal_distances = np.linalg.norm(state.pos - state.goals, axis=1)
        info = {
            "goal_distance": goal_distances,
            "termination_reason": None,
            "collisions": [],
        }

        if self.logger:
            metadata = {"seed": seed, "randomize": randomize}
            self.logger.start_episode(self._episode_num, metadata)

        self._episode_reward = 0.0
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.num_drones, 3):
            raise ValueError(f"Expected action shape ({self.num_drones}, 3), got {action.shape}")

        clipped_action = np.clip(action, -self._action_limit, self._action_limit)
        state, sim_info = self.simulator.step(clipped_action)

        obs = self._build_observation(state, sim_info)
        rewards = self.reward_fn.compute(state, clipped_action, sim_info)

        terminated = bool(sim_info["done"]) and (
            sim_info["termination_reason"] != "timeout"
        )
        truncated = bool(sim_info["done"]) and (
            sim_info["termination_reason"] == "timeout"
        )

        goal_distances = np.linalg.norm(state.pos - state.goals, axis=1)
        info = {
            "goal_distance": goal_distances,
            "termination_reason": sim_info["termination_reason"],
            "collisions": sim_info["collisions"],
        }

        self._episode_reward += rewards

        # End episode logging and save simulation run
        if (terminated or truncated) and self.logger:
            result = self.logger.end_episode(
                termination_reason=sim_info["termination_reason"],
                episode_stats=sim_info["episode_stats"],
                total_reward=float(np.sum(self._episode_reward)),  # Log sum of all drone rewards
            )
            info["episode_result"] = result  # Add to info dict
            
            # Save simulation run for visualization if enabled (use current episode num before incrementing)
            if self._save_runs and self.simulator.current_run and self.simulator.current_run.frames:
                self._save_episode_run(self._episode_num)
            
            self._episode_num += 1

        return obs, rewards, terminated, truncated, info

    def save_episode_logs(self):
        """
        Save the collection of episode logs to disk.

        Call this when you want to checkpoint episode results (e.g., after training steps or at the end).
        Does nothing if log_dir was not specified.
        """
        if self.logger:
            self.logger.save_to_disk()
    
    def _save_episode_run(self, episode_num: Optional[int] = None):
        """Save the current simulation run to disk."""
        if not self.logger or not self.logger.log_dir:
            return
        
        # Use provided episode number or current one
        if episode_num is None:
            episode_num = self._episode_num
        
        # Add obstacles and environment info to metadata
        if self.simulator.current_run:
            if "environment" not in self.simulator.current_run.metadata:
                self.simulator.current_run.metadata["environment"] = {}
            
            # Store obstacle objects directly
            self.simulator.current_run.metadata["environment"]["obstacles"] = (
                self.environment.obstacles
            )
            self.simulator.current_run.metadata["environment"]["bounds"] = (
                list(self.environment.bounds)
            )
            self.simulator.current_run.metadata["environment"]["start_position"] = (
                list(self.environment.start_position)
            )
            self.simulator.current_run.metadata["environment"]["goal_position"] = (
                list(self.environment.goal_position)
            )
            
            # Store entire config for visualization and reproducibility
            self.simulator.current_run.metadata["config"] = self.config
            
            # Save to same directory as episode results
            output_path = self.logger.log_dir / f"episode_{episode_num:06d}.json"
            self.simulator.save_run(output_path)

    def render(self) -> None:
        # Offline rendering is handled by the simulator's logger/visualizer.
        return None

    def close(self) -> None:
        return None
