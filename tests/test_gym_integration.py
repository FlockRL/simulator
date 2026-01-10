"""Tests for FlockRLGymEnv integration with episode logging."""

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest
import yaml

from flockrl_sim import FlockRLGymEnv
from flockrl_sim.environment import Environment
from flockrl_sim.rewards import RewardFunction
from flockrl_sim.state import SwarmState
from flockrl_sim.gym_logging import EpisodeResult


def create_test_env(
    reward_fn,
    log_dir: Optional[Path] = None,
    **kwargs
):
    """Helper to create test environment with defaults."""
    environment = kwargs.pop(
        "environment",
        Environment(
            bounds=(-100, 100, -100, 100, 0, 100),
            obstacles=[],
            start_position=(0.0, 0.0, 1.0),
            goal_position=(0.0, 0.0, 10.0),
            seed=0,
        ),
    )
    
    # Create temporary config file if log_dir is provided
    config_path = None
    if log_dir is not None:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            config = {
                "simulation": {
                    "delta_t": 0.004166666666666667,
                    "max_steps": 1000,
                    "goal_threshold": 0.5,
                    "max_acceleration": 5.0,
                    "terminate_on_collision": True,
                },
                "gym": {
                    "max_neighbors": 4,
                    "log_dir": str(log_dir),
                },
                "collision": {
                    "restitution": 0.8,
                    "enable_collisions": True,
                },
            }
            yaml.dump(config, f)
            config_path = Path(f.name)
    
    return FlockRLGymEnv(
        reward_fn=reward_fn,
        environment=environment,
        config_path=config_path,
    )


class SimpleReward(RewardFunction):
    """Simple reward function for testing."""

    def reset(self, state: SwarmState) -> None:
        self._last_dist = float(np.linalg.norm(state.pos[0] - state.goals[0]))

    def compute(
        self, state: SwarmState, action: np.ndarray, sim_info: Dict[str, Any]
    ) -> float:
        current_dist = float(np.linalg.norm(state.pos[0] - state.goals[0]))
        reward = self._last_dist - current_dist
        if sim_info.get("termination_reason") == "success":
            reward += 100.0
        self._last_dist = current_dist
        return reward


class TestFlockRLGymEnvWithoutLogging:
    """Test FlockRLGymEnv with logging disabled (backward compatibility)."""

    def test_env_without_logging(self):
        """Test that environment works without logging enabled."""
        env = create_test_env(SimpleReward())

        assert env.logger is None
        assert env._episode_num == 0
        assert env._episode_reward == 0.0

    def test_reset_without_logging(self):
        """Test reset works without logging."""
        env = create_test_env(SimpleReward())
        obs, info = env.reset()

        assert obs.shape == env.observation_space.shape
        assert "goal_distance" in info
        assert "termination_reason" in info

    def test_step_without_logging(self):
        """Test step works without logging."""
        env = create_test_env(SimpleReward())
        env.reset()

        action = np.array([1.0, 0.0, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "episode_result" not in info  # No logging

    def test_logger_access_requires_logging(self):
        """Test that accessing logger requires logging to be enabled."""
        env = create_test_env(SimpleReward())
        assert env.logger is None


class TestFlockRLGymEnvWithLogging:
    """Test FlockRLGymEnv with logging enabled."""

    def test_env_with_logging(self):
        """Test environment initialization with logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = create_test_env(
                SimpleReward(), log_dir=Path(tmpdir)
            )

            assert env.logger is not None
            assert env.logger.log_dir == Path(tmpdir)

    def test_reset_starts_episode_logging(self):
        """Test that reset starts episode logging."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = create_test_env(
                SimpleReward(), log_dir=Path(tmpdir)
            )
        env.reset(seed=42)

        assert env.logger._current_episode_num == 0
        assert env.logger._current_episode_metadata == {"seed": 42}

    def test_episode_logging_during_training(self):
        """Test episode logging during a complete training episode."""
        # Use short max_steps to ensure episode completes quickly
        with tempfile.TemporaryDirectory() as tmpdir:
            env = create_test_env(
                SimpleReward(),
                log_dir=Path(tmpdir),
            )

        # Run one episode to completion
        obs, info = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Check that episode was logged
        results = list(env.logger._results)
        assert len(results) == 1
        assert results[0].episode_num == 0
        assert results[0].steps > 0

    def test_episode_result_in_info_dict(self):
        """Test that episode_result is added to info dict on episode end."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = create_test_env(
                SimpleReward(),
                log_dir=Path(tmpdir),
            )

            obs, info = env.reset()
            done = False

            # Run until episode ends
            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            # Episode result should be in final info dict
            assert "episode_result" in info
            assert isinstance(info["episode_result"], EpisodeResult)
            assert info["episode_result"].episode_num == 0

    def test_multiple_episodes_logging(self):
        """Test logging across multiple episodes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = create_test_env(
                SimpleReward(),
                log_dir=Path(tmpdir),
            )

            num_episodes = 5

            for episode in range(num_episodes):
                obs, info = env.reset()
                done = False

                while not done:
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

            # Check all episodes were logged
            results = list(env.logger._results)
            assert len(results) == num_episodes
            assert [r.episode_num for r in results] == list(range(num_episodes))

    def test_custom_stats_calculation(self):
        """Test that users can calculate stats directly from logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = create_test_env(
                SimpleReward(),
                log_dir=Path(tmpdir),
            )

            # Run a few episodes
            for _ in range(10):
                obs, info = env.reset()
                done = False

                while not done:
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

            # Users can calculate stats directly
            results = list(env.logger._results)[-10:]
            if results:
                successes = sum(1 for r in results if r.termination_reason == "success")
                collisions = sum(1 for r in results if r.termination_reason == "collision")
                success_rate = successes / len(results)
                collision_rate = collisions / len(results)
                
                assert isinstance(success_rate, float)
                assert 0.0 <= success_rate <= 1.0
                assert isinstance(collision_rate, float)
                assert 0.0 <= collision_rate <= 1.0

    def test_logger_summary_stats(self):
        """Test accessing summary stats from logger."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = create_test_env(
                SimpleReward(),
                log_dir=Path(tmpdir),
            )

            # Run a few episodes
            for _ in range(3):
                obs, info = env.reset()
                done = False

                while not done:
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

            # Compute summary stats directly
            results = list(env.logger._results)
            total = len(results)
            successes = sum(1 for r in results if r.termination_reason == "success")
            collisions = sum(1 for r in results if r.termination_reason == "collision")
            avg_steps = sum(r.steps for r in results) / total if results else 0
            avg_reward = sum(r.total_reward for r in results) / total if results else 0

            assert total == 3
            assert successes >= 0
            assert collisions >= 0
            assert avg_steps >= 0
            assert np.isfinite(avg_reward)

    def test_save_logs(self):
        """Test manual save_logs method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = create_test_env(
                SimpleReward(),
                log_dir=Path(tmpdir),
            )

            # Run one episode
            obs, info = env.reset()
            done = False

            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            # Manually save logs
            env.save_logs()

            # Check that file was created
            json_path = Path(tmpdir) / "episode_results.json"
            assert json_path.exists()

    def test_simulation_run_saving(self):
        """Test that simulation runs are saved when episodes end."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env = create_test_env(
                SimpleReward(),
                log_dir=Path(tmpdir),
            )

            # Run one episode to completion
            obs, info = env.reset()
            done = False
            step_count = 0

            while not done:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                step_count += 1

            # Check that simulation run was saved
            episode_json = Path(tmpdir) / "episode_000000.json"
            assert episode_json.exists()
            
            # Verify it's valid JSON with frames
            import json
            with open(episode_json) as f:
                data = json.load(f)
            assert "frames" in data
            assert len(data["frames"]) == step_count + 1  # +1 for initial frame
            assert "metadata" in data
            assert "environment" in data["metadata"]

    def test_logger_dataframe(self):
        """Test accessing DataFrame from logger."""
        pytest.importorskip("pandas")

        with tempfile.TemporaryDirectory() as tmpdir:
            env = create_test_env(
                SimpleReward(),
                log_dir=Path(tmpdir),
            )

            # Run a few episodes
            for _ in range(5):
                obs, info = env.reset()
                done = False

                while not done:
                    action = env.action_space.sample()
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

            # Create DataFrame directly from results
            import pandas as pd
            from dataclasses import asdict
            
            results = list(env.logger._results)[-5:]
            data = []
            for r in results:
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

        assert len(df) == 5
        assert "episode_num" in df.columns
        assert "termination_reason" in df.columns
        assert "total_reward" in df.columns


class TestRewardFunctions:
    """Test custom reward functions."""

    def test_custom_reward_function_subclass(self):
        """Test that users can define custom reward functions."""

        class ConstantReward(RewardFunction):
            """Simple constant reward for testing."""

            def __init__(self, constant_value: float = 1.0):
                self.constant_value = constant_value

            def reset(self, state: SwarmState) -> None:
                pass

            def compute(
                self, state: SwarmState, action: np.ndarray, sim_info: Dict[str, Any]
            ) -> float:
                return self.constant_value

        reward_fn = ConstantReward(constant_value=5.0)
        env = create_test_env(reward_fn)

        obs, info = env.reset()
        action = np.array([1.0, 0.0, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)

        # Should get constant reward
        assert reward == 5.0

    def test_reward_function_reset_called(self):
        """Test that reward function reset is called on env reset."""

        class TrackingReward(RewardFunction):
            """Reward function that tracks reset calls."""

            def __init__(self):
                self.reset_count = 0

            def reset(self, state: SwarmState) -> None:
                self.reset_count += 1

            def compute(
                self, state: SwarmState, action: np.ndarray, sim_info: Dict[str, Any]
            ) -> float:
                return 0.0

        reward_fn = TrackingReward()
        env = create_test_env(reward_fn)

        assert reward_fn.reset_count == 0

        env.reset()
        assert reward_fn.reset_count == 1

        env.reset()
        assert reward_fn.reset_count == 2

    def test_reward_function_required(self):
        """Test that reward_fn is required."""
        with pytest.raises(TypeError):
            FlockRLGymEnv()  # Should fail without reward_fn

    def test_reset_optional(self):
        """Test that reset() can be omitted if not needed."""
        # Reward function that doesn't need state tracking
        class StatelessReward(RewardFunction):
            """Reward function that doesn't need reset()."""

            def compute(
                self, state: SwarmState, action: np.ndarray, sim_info: Dict[str, Any]
            ) -> float:
                # Just return a constant reward
                return 1.0

        reward_fn = StatelessReward()
        env = create_test_env(reward_fn)

        # Should work fine - reset() has default no-op implementation
        obs, info = env.reset()
        action = np.array([1.0, 0.0, 0.0])
        obs, reward, terminated, truncated, info = env.step(action)

        assert reward == 1.0
