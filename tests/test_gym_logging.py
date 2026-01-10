"""Tests for episode logging functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from flockrl_sim.gym_logging import EpisodeLogger, EpisodeResult, TrajectoryData


class TestEpisodeResult:
    """Test EpisodeResult dataclass."""

    def test_duration_property(self):
        """Test duration property calculation."""
        result = EpisodeResult(
            episode_num=0,
            steps=100,
            termination_reason="success",
            total_reward=50.0,
            final_goal_distance=0.3,
            min_goal_distance=0.2,
            collision_count=0,
            start_time=1000.0,
            end_time=1010.5,
        )
        assert result.duration == 10.5


class TestEpisodeLogger:
    """Test EpisodeLogger class."""

    def test_init_no_log_dir(self):
        """Test logger initialization without log directory."""
        logger = EpisodeLogger()
        assert logger.log_dir is None
        assert not logger._save_trajectories
        assert len(logger._results) == 0

    def test_init_with_log_dir(self):
        """Test logger initialization with log directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = EpisodeLogger(log_dir=Path(tmpdir))
            assert logger.log_dir == Path(tmpdir)
            assert logger.log_dir.exists()

    def test_start_episode(self):
        """Test starting episode tracking."""
        logger = EpisodeLogger()
        metadata = {"seed": 42, "test": "value"}
        logger.start_episode(episode_num=0, metadata=metadata)

        assert logger._current_episode_num == 0
        assert logger._current_episode_metadata == metadata
        assert logger._current_episode_start_time is not None

    def test_log_step_without_trajectory_saving(self):
        """Test that log_step does nothing when save_trajectories=False."""
        logger = EpisodeLogger(save_trajectories=False)
        logger.start_episode(0)

        position = np.array([1.0, 2.0, 3.0])
        action = np.array([0.1, 0.2, 0.3])
        logger.log_step(position, action, reward=1.0, timestep=0.1)

        # Should not create trajectory buffer
        assert logger._current_trajectory_buffer is None

    def test_log_step_with_trajectory_saving(self):
        """Test step logging when save_trajectories=True."""
        logger = EpisodeLogger(save_trajectories=True)
        logger.start_episode(0)

        position = np.array([1.0, 2.0, 3.0])
        action = np.array([0.1, 0.2, 0.3])
        logger.log_step(position, action, reward=1.5, timestep=0.1)

        assert logger._current_trajectory_buffer is not None
        assert len(logger._current_trajectory_buffer["positions"]) == 1
        assert len(logger._current_trajectory_buffer["actions"]) == 1
        assert logger._current_trajectory_buffer["rewards"][0] == 1.5

    def test_end_episode(self):
        """Test ending episode and creating result."""
        logger = EpisodeLogger()
        logger.start_episode(0, metadata={"seed": 42})

        episode_stats = {
            "total_steps": 150,
            "final_goal_distance": 0.4,
            "min_goal_distance": 0.25,
            "collision_count": 2,
        }

        result = logger.end_episode(
            termination_reason="success",
            episode_stats=episode_stats,
            total_reward=75.5,
        )

        assert isinstance(result, EpisodeResult)
        assert result.episode_num == 0
        assert result.steps == 150
        assert result.termination_reason == "success"
        assert result.total_reward == 75.5
        assert result.final_goal_distance == 0.4
        assert len(logger._results) == 1

    def test_end_episode_with_trajectories(self):
        """Test ending episode with trajectory saving."""
        logger = EpisodeLogger(save_trajectories=True)
        logger.start_episode(0)

        # Log some steps
        for i in range(5):
            pos = np.array([float(i), 0.0, 0.0])
            action = np.array([1.0, 0.0, 0.0])
            logger.log_step(pos, action, reward=1.0, timestep=float(i))

        episode_stats = {
            "total_steps": 5,
            "final_goal_distance": 5.0,
            "min_goal_distance": 0.0,
            "collision_count": 0,
        }

        result = logger.end_episode("success", episode_stats, total_reward=5.0)

        # Check trajectory was saved
        assert 0 in logger._trajectories
        trajectory = logger._trajectories[0]
        assert isinstance(trajectory, TrajectoryData)
        assert trajectory.positions.shape == (5, 3)
        assert trajectory.actions.shape == (5, 3)
        assert trajectory.rewards.shape == (5,)

    def test_results_access(self):
        """Test accessing episode results directly."""
        logger = EpisodeLogger()

        # Log 10 episodes
        for i in range(10):
            logger.start_episode(i)
            stats = {
                "total_steps": 100,
                "final_goal_distance": 1.0,
                "min_goal_distance": 0.5,
                "collision_count": 0,
            }
            logger.end_episode("success", stats, total_reward=50.0)

        # Get all results
        all_results = list(logger._results)
        assert len(all_results) == 10

        # Get last 5
        last_5 = list(logger._results)[-5:]
        assert len(last_5) == 5
        assert last_5[0].episode_num == 5


    def test_compute_summary_stats_directly(self):
        """Test computing summary statistics directly from results."""
        logger = EpisodeLogger()

        # Log episodes with different outcomes
        outcomes = ["success", "success", "collision", "timeout", "success"]
        for i, outcome in enumerate(outcomes):
            logger.start_episode(i)
            stats = {
                "total_steps": 100 + i * 10,
                "final_goal_distance": float(i),
                "min_goal_distance": float(i) * 0.5,
                "collision_count": 1 if outcome == "collision" else 0,
            }
            logger.end_episode(outcome, stats, total_reward=50.0 + i * 10)

        # Compute stats directly
        results = list(logger._results)
        total = len(results)
        successes = sum(1 for r in results if r.termination_reason == "success")
        collisions = sum(1 for r in results if r.termination_reason == "collision")
        timeouts = sum(1 for r in results if r.termination_reason == "timeout")
        avg_steps = sum(r.steps for r in results) / total

        assert total == 5
        assert successes == 3  # 3 successes out of 5
        assert collisions == 1
        assert timeouts == 1
        assert avg_steps == (100 + 110 + 120 + 130 + 140) / 5

    def test_save_to_disk_json(self):
        """Test saving episode results to JSON."""
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = EpisodeLogger(log_dir=Path(tmpdir))

            # Log an episode
            logger.start_episode(0)
            stats = {
                "total_steps": 100,
                "final_goal_distance": 0.5,
                "min_goal_distance": 0.2,
                "collision_count": 0,
            }
            logger.end_episode("success", stats, total_reward=75.0)

            # Save to disk
            logger.save_to_disk(force=True)

            # Check file was created
            json_path = Path(tmpdir) / "episode_results.json"
            assert json_path.exists()

            # Verify contents
            with open(json_path) as f:
                data = json.load(f)
            assert len(data) == 1
            assert data[0]["episode_num"] == 0
            assert data[0]["termination_reason"] == "success"

    def test_save_trajectories_npz(self):
        """Test saving trajectories to NPZ format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = EpisodeLogger(
                log_dir=Path(tmpdir), save_trajectories=True
            )

            # Log episode with trajectory
            logger.start_episode(0)
            for i in range(5):
                pos = np.array([float(i), 0.0, 0.0])
                action = np.array([1.0, 0.0, 0.0])
                logger.log_step(pos, action, reward=1.0, timestep=float(i))

            stats = {
                "total_steps": 5,
                "final_goal_distance": 5.0,
                "min_goal_distance": 0.0,
                "collision_count": 0,
            }
            logger.end_episode("success", stats, total_reward=5.0)

            # Save to disk
            logger.save_to_disk(force=True)

            # Check trajectory file was created
            traj_path = Path(tmpdir) / "trajectories" / "episode_000000.npz"
            assert traj_path.exists()

            # Verify contents
            data = np.load(traj_path)
            assert "positions" in data
            assert "actions" in data
            assert data["positions"].shape == (5, 3)

    def test_create_dataframe_directly(self):
        """Test creating DataFrame directly from results."""
        pytest.importorskip("pandas")
        from dataclasses import asdict

        logger = EpisodeLogger()

        # Log some episodes
        for i in range(3):
            logger.start_episode(i)
            stats = {
                "total_steps": 100,
                "final_goal_distance": 1.0,
                "min_goal_distance": 0.5,
                "collision_count": 0,
            }
            logger.end_episode("success", stats, total_reward=50.0)

        # Create DataFrame directly
        import pandas as pd
        data = []
        for r in logger._results:
            row = {
                "episode_num": r.episode_num,
                "steps": r.steps,
                "termination_reason": r.termination_reason,
                "total_reward": r.total_reward,
                "final_goal_distance": r.final_goal_distance,
                "min_goal_distance": r.min_goal_distance,
                "collision_count": r.collision_count,
                "duration": r.duration,
            }
            data.append(row)
        
        df = pd.DataFrame(data)

        assert len(df) == 3
        assert "episode_num" in df.columns
        assert "termination_reason" in df.columns
        assert "total_reward" in df.columns
        assert df["episode_num"].tolist() == [0, 1, 2]
