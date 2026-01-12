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
            steps=episode_stats["total_steps"],
            termination_reason=termination_reason,
            total_reward=total_reward,
            final_goal_distance=episode_stats["final_goal_distance"],
            min_goal_distance=episode_stats["min_goal_distance"],
            collision_count=episode_stats["collision_count"],
            start_time=self._current_episode_start_time,
            end_time=time.time(),
            metadata=self._current_episode_metadata,
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
        """Save episode results as JSON (human-readable), appending to existing file if it exists."""
        if not self.log_dir:
            return

        if not self._results:
            return

        output_path = self.log_dir / "episode_results.json"
        
        # Load existing results if file exists
        existing_results = []
        if output_path.exists():
            try:
                with open(output_path, "r") as f:
                    existing_results = json.load(f)
            except (json.JSONDecodeError, IOError):
                # If file is corrupted or unreadable, start fresh
                existing_results = []
        
        # Convert existing results to dict format for comparison
        existing_dict = {(r["episode_num"], r["start_time"]): r for r in existing_results}
        
        # Add new results, replacing any duplicates (same episode_num + start_time)
        new_results_dict = [asdict(r) for r in self._results]
        for new_result in new_results_dict:
            key = (new_result["episode_num"], new_result["start_time"])
            existing_dict[key] = new_result
        
        # Convert back to list and sort by episode_num, then start_time
        all_results = list(existing_dict.values())
        all_results.sort(key=lambda x: (x["episode_num"], x["start_time"]))
        
        # Save combined results
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
