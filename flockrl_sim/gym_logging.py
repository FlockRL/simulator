"""
Episode logging system for FlockRL Gymnasium environment.

Provides lightweight episode outcome tracking with optional trajectory storage.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import numpy as np


@dataclass
class EpisodeResult:
    """Lightweight episode outcome record."""

    episode_num: int
    steps: int
    termination_reason: str  # "success", "collision", "timeout", "out_of_bounds"
    total_reward: float
    final_goal_distance: float
    min_goal_distance: float
    collision_count: int
    start_time: float
    end_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Get episode duration in seconds."""
        return self.end_time - self.start_time


@dataclass
class TrajectoryData:
    """Optional full trajectory storage for detailed analysis."""

    episode_num: int
    positions: np.ndarray  # (T, 3) positions
    actions: np.ndarray  # (T, 3) actions
    rewards: np.ndarray  # (T,) rewards
    timestamps: np.ndarray  # (T,) timesteps


class EpisodeLogger:
    """
    Track episode outcomes and optionally save trajectories.

    Features:
    - Manual save: call save_to_disk() when you want to checkpoint
    - Optional trajectory storage (disabled by default)
    - Episode results saved as JSON (human-readable)
    - Trajectories saved as NPZ (fast, efficient for large arrays)
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
        save_trajectories: bool = False,
    ):
        """
        Args:
            log_dir: Directory for saving logs. If None, logs only kept in memory.
            save_trajectories: Whether to save full position/action trajectories.
        """
        self.log_dir = Path(log_dir) if log_dir else None
        self._save_trajectories = save_trajectories

        # Create log directory if specified
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self._results: List[EpisodeResult] = []

        # Trajectory storage (only if enabled)
        self._trajectories: Dict[int, TrajectoryData] = {}

        # Current episode tracking
        self._current_episode_num: Optional[int] = None
        self._current_episode_start_time: Optional[float] = None
        self._current_episode_metadata: Optional[Dict] = None
        self._current_trajectory_buffer: Optional[Dict[str, List]] = None

        # Total episodes processed (for save interval)
        self._total_episodes = 0

    def start_episode(
        self, episode_num: int, metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Begin tracking a new episode.

        Args:
            episode_num: Episode number
            metadata: Optional metadata to store with episode result
        """
        self._current_episode_num = episode_num
        self._current_episode_start_time = time.time()
        self._current_episode_metadata = metadata or {}

        # Initialize trajectory buffer if needed
        if self._save_trajectories:
            self._current_trajectory_buffer = {
                "positions": [],
                "actions": [],
                "rewards": [],
                "timestamps": [],
            }

    def log_step(
        self,
        position: np.ndarray,
        action: np.ndarray,
        reward: float,
        timestep: float,
    ):
        """
        Log data for current step (only if save_trajectories=True).

        Args:
            position: Drone position (3,)
            action: Action taken (3,)
            reward: Reward received
            timestep: Simulation time
        """
        if not self._save_trajectories or self._current_trajectory_buffer is None:
            return

        self._current_trajectory_buffer["positions"].append(position.copy())
        self._current_trajectory_buffer["actions"].append(action.copy())
        self._current_trajectory_buffer["rewards"].append(reward)
        self._current_trajectory_buffer["timestamps"].append(timestep)

    def end_episode(
        self,
        termination_reason: str,
        episode_stats: Dict[str, Any],
        total_reward: float,
    ) -> EpisodeResult:
        """
        Finalize episode and return result.

        Args:
            termination_reason: How the episode ended
            episode_stats: Episode statistics from simulator
            total_reward: Cumulative reward over episode

        Returns:
            EpisodeResult for this episode
        """
        if self._current_episode_num is None:
            raise RuntimeError("end_episode() called without start_episode()")

        # Create episode result
        result = EpisodeResult(
            episode_num=self._current_episode_num,
            steps=episode_stats.get("total_steps", 0),
            termination_reason=termination_reason or "unknown",
            total_reward=total_reward,
            final_goal_distance=episode_stats.get("final_goal_distance", float("inf")),
            min_goal_distance=episode_stats.get("min_goal_distance", float("inf")),
            collision_count=episode_stats.get("collision_count", 0),
            start_time=self._current_episode_start_time or 0.0,
            end_time=time.time(),
            metadata=self._current_episode_metadata or {},
        )

        # Store result
        self._results.append(result)

        # Store trajectory if enabled
        if self._save_trajectories and self._current_trajectory_buffer:
            trajectory = TrajectoryData(
                episode_num=self._current_episode_num,
                positions=np.array(self._current_trajectory_buffer["positions"]),
                actions=np.array(self._current_trajectory_buffer["actions"]),
                rewards=np.array(self._current_trajectory_buffer["rewards"]),
                timestamps=np.array(self._current_trajectory_buffer["timestamps"]),
            )
            self._trajectories[self._current_episode_num] = trajectory

        # Increment total episodes counter
        self._total_episodes += 1

        # Reset current episode tracking
        self._current_episode_num = None
        self._current_episode_start_time = None
        self._current_episode_metadata = None
        self._current_trajectory_buffer = None

        return result

    def save_to_disk(self):
        """Save accumulated results to disk."""
        # Save episode results as JSON (human-readable)
        self._save_results_json()

        # Save trajectories if enabled
        if self._save_trajectories and self._trajectories:
            self._save_trajectories_npz()

    def _save_results_json(self, force: bool = False):
        """Save episode results as JSON (human-readable)."""
        if not self.log_dir:
            return

        if not self._results:
            return

        results_dict = [asdict(r) for r in self._results]
        output_path = self.log_dir / "episode_results.json"

        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2)

    def _save_trajectories_npz(self, force: bool = False):
        """Save trajectory data as compressed numpy arrays."""
        if not self.log_dir:
            return

        if not self._trajectories:
            return

        trajectories_dir = self.log_dir / "trajectories"
        trajectories_dir.mkdir(exist_ok=True)

        for episode_num, traj in self._trajectories.items():
            output_path = trajectories_dir / f"episode_{episode_num:06d}.npz"
            np.savez_compressed(
                output_path,
                positions=traj.positions,
                actions=traj.actions,
                rewards=traj.rewards,
                timestamps=traj.timestamps,
            )
