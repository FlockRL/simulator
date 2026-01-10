"""
Episode logging system for FlockRL Gymnasium environment.

Provides lightweight episode outcome tracking.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import json


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


class EpisodeLogger:
    """
    Track episode outcomes.

    Features:
    - Manual save: call save_to_disk() when you want to checkpoint
    - Episode results saved as JSON (human-readable)
    """

    def __init__(
        self,
        log_dir: Optional[Path] = None,
    ):
        """
        Args:
            log_dir: Directory for saving logs. If None, logs only kept in memory.
        """
        self.log_dir = Path(log_dir) if log_dir else None

        # Create log directory if specified
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # Results storage
        self._results: List[EpisodeResult] = []

        # Current episode tracking
        self._current_episode_num: Optional[int] = None
        self._current_episode_start_time: Optional[float] = None
        self._current_episode_metadata: Optional[Dict] = None

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

        # Increment total episodes counter
        self._total_episodes += 1

        # Reset current episode tracking
        self._current_episode_num = None
        self._current_episode_start_time = None
        self._current_episode_metadata = None

        return result

    def save_to_disk(self):
        """Save accumulated results to disk."""
        # Save episode results as JSON (human-readable)
        self._save_results_json()

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
