"""
Gymnasium environment wrapper around the CoreSimulator.

This environment makes it easy to plug the existing simulator into standard
RL tooling that expects the Gymnasium API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import yaml
from gymnasium import spaces

from .collision.system import CollisionSystem
from .environment import Environment
from .gym_logging import EpisodeLogger
from .rewards import RewardFunction
from .simulator import CoreSimulator
from .state import SwarmState


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        # Look for config.yml in the project root (parent of flockrl_sim)
        config_path = Path(__file__).parent.parent / "config.yml"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}. "
            "Please create config.yml in the project root."
        )
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}
    
    return config


class FlockRLGymEnv(gym.Env):
    """Single-drone Gymnasium environment backed by the CoreSimulator."""

    metadata = {"render_modes": ["none"], "render_fps": 60}

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
        
        self.environment = environment
        self.sim_config = sim_config
        
        # Create collision system
        collision_system = self._build_collision_system(collision_config)
        
        # Whether to save simulation runs for visualization
        self._save_runs = gym_config["save_runs"]
        
        self.simulator = CoreSimulator(
            delta_t=sim_config["delta_t"],
            max_steps=sim_config["max_steps"],
            goal_threshold=sim_config["goal_threshold"],
            max_acceleration=sim_config["max_acceleration"],
            terminate_on_collision=sim_config["terminate_on_collision"],
            collision_system=collision_system,
            environment=self.environment,
            enable_frame_logging=self._save_runs,  # Only log frames if we'll save them
        )

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
        log_dir = gym_config["log_dir"]
        if log_dir:
            self.logger = EpisodeLogger(log_dir=Path(log_dir))

        self._episode_num = 0
        self._episode_reward = 0.0

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

    def _build_collision_system(
        self, collision_config: Dict[str, Any]
    ) -> Optional[CollisionSystem]:
        """Create a CollisionSystem instance when enabled in config."""
        enable = collision_config.get("enable_collisions", True)
        if not enable:
            return None

        restitution = collision_config.get("restitution", 0.8)
        return CollisionSystem(environment=self.environment, restitution=restitution)

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
        goal_distance = float(np.linalg.norm(goal_vector))
        vel = (
            state.vel[0].astype(np.float32)
            if state.vel is not None
            else np.zeros(3, dtype=np.float32)
        )

        obs_parts = [
            state.pos[0].astype(np.float32),
            vel,
            goal_vector,
            np.array([goal_distance], dtype=np.float32),
            ranges,
            hits,
            neighbor_vectors.flatten(),
        ]
        return np.concatenate(obs_parts, dtype=np.float32)

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
        
        # Initialize reward function
        self.reward_fn.reset(state)
        
        obs = self._build_observation(state)
        goal_distance = float(np.linalg.norm(state.pos[0] - state.goals[0]))
        info = {
            "goal_distance": goal_distance,
            "termination_reason": None,
            "collisions": [],
        }

        # Start episode logging
        if self.logger:
            metadata = {"seed": seed, **(options or {})}
            self.logger.start_episode(self._episode_num, metadata)

        self._episode_reward = 0.0

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
        reward = self.reward_fn.compute(state, clipped_action, sim_info)

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

        self._episode_reward += reward

        # End episode logging and save simulation run
        if (terminated or truncated) and self.logger:
            result = self.logger.end_episode(
                termination_reason=sim_info.get("termination_reason"),
                episode_stats=sim_info["episode_stats"],
                total_reward=self._episode_reward,
            )
            info["episode_result"] = result  # Add to info dict
            
            # Save simulation run for visualization if enabled (use current episode num before incrementing)
            if self._save_runs and self.simulator.current_run and self.simulator.current_run.frames:
                self._save_episode_run(self._episode_num)
            
            self._episode_num += 1

        return obs, reward, terminated, truncated, info

    def save_logs(self):
        """
        Manually trigger save of episode logs to disk.

        Call this when you want to checkpoint (e.g., after training steps or at the end).
        Does nothing if log_dir was not specified.
        """
        if self.logger:
            self.logger.save_to_disk()
            
            # Also save current simulation run if enabled and it exists
            if self._save_runs and self.simulator.current_run and self.simulator.current_run.frames:
                self._save_episode_run()
    
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
            
            # Save to same directory as episode results
            output_path = self.logger.log_dir / f"episode_{episode_num:06d}.json"
            self.simulator.save_run(output_path)

    def render(self) -> None:
        # Offline rendering is handled by the simulator's logger/visualizer.
        return None

    def close(self) -> None:
        return None
