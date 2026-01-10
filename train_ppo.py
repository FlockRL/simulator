"""
PPO Training Script for FlockRL Drone Navigation

This script trains a PPO agent to navigate through gates and obstacles to reach a goal.
Uses Stable-Baselines3 for the PPO implementation.

Configuration is defined in the TRAINING_CONFIG dictionary below.
Edit the values directly in this file to customize training.

Usage:
    python train_ppo.py
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple
import csv

import gymnasium as gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecNormalize

from flockrl_sim import FlockRLGymEnv, SimulationConfig
from flockrl_sim.environment import EnvironmentSpecLoader, EnvironmentBuilder


class TimeoutVecEnv(VecEnv):
    """
    Wrapper around VecEnv that adds timeout protection to prevent deadlocks.
    
    This is especially useful for SubprocVecEnv where subprocesses can hang.
    Uses threading to wrap step/reset calls with timeouts.
    """
    
    def __init__(self, venv: VecEnv, timeout_seconds: float = 30.0):
        """
        Args:
            venv: The VecEnv to wrap
            timeout_seconds: Maximum time to wait for step/reset operations
        """
        super().__init__(
            venv.num_envs,
            venv.observation_space,
            venv.action_space,
        )
        self.venv = venv
        self.timeout_seconds = timeout_seconds
        self.timeout_count = 0
        self.executor = ThreadPoolExecutor(max_workers=1)
    
    def step_async(self, actions: np.ndarray) -> None:
        """Async step with timeout protection."""
        self.venv.step_async(actions)
    
    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """Wait for step with timeout protection."""
        future = self.executor.submit(self.venv.step_wait)
        try:
            return future.result(timeout=self.timeout_seconds)
        except TimeoutError:
            self.timeout_count += 1
            print(f"WARNING: step_wait() timed out after {self.timeout_seconds}s (timeout #{self.timeout_count})")
            # Try to reset all environments to recover
            try:
                obs = self.reset()
                # Return done=True for all envs to signal timeout
                dones = np.ones(self.num_envs, dtype=bool)
                rewards = np.zeros(self.num_envs, dtype=np.float32)
                infos = [{"timeout": True} for _ in range(self.num_envs)]
                return obs, rewards, dones, dones.copy(), infos
            except Exception as e:
                print(f"ERROR: Failed to recover from timeout: {e}")
                raise
    
    def reset(self) -> np.ndarray:
        """Reset with timeout protection."""
        future = self.executor.submit(self.venv.reset)
        try:
            return future.result(timeout=self.timeout_seconds)
        except TimeoutError:
            self.timeout_count += 1
            print(f"WARNING: reset() timed out after {self.timeout_seconds}s (timeout #{self.timeout_count})")
            # Try again once
            future = self.executor.submit(self.venv.reset)
            try:
                return future.result(timeout=self.timeout_seconds * 2)
            except TimeoutError:
                raise RuntimeError(f"reset() timed out twice. Subprocess may be deadlocked.")
    
    def close(self) -> None:
        """Close the wrapped environment."""
        self.executor.shutdown(wait=False)
        self.venv.close()
    
    def get_attr(self, attr_name: str, indices=None):
        """Get attribute from wrapped environment."""
        return self.venv.get_attr(attr_name, indices)
    
    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        """Set attribute in wrapped environment."""
        self.venv.set_attr(attr_name, value, indices)
    
    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        """Call method on wrapped environment."""
        return self.venv.env_method(method_name, *method_args, indices=indices, **method_kwargs)
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        """Check if environment is wrapped by a specific wrapper class."""
        return self.venv.env_is_wrapped(wrapper_class, indices=indices)
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """Step with timeout protection."""
        self.step_async(actions)
        return self.step_wait()


class TimeoutWrapper(gym.Wrapper):
    """
    Wrapper that forces environment reset if episode runs too long (prevents stuck envs).
    
    This is useful for SubprocVecEnv where environments can hang and cause deadlocks.
    """
    
    def __init__(self, env: gym.Env, timeout_seconds: float = 60.0):
        """
        Args:
            env: The environment to wrap
            timeout_seconds: Maximum real-world time (seconds) an episode can run before forced reset
        """
        super().__init__(env)
        self.timeout_seconds = timeout_seconds
        self.episode_start_time: float = 0.0
        self.forced_resets: int = 0
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment and record start time."""
        self.episode_start_time = time.time()
        return self.env.reset(**kwargs)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Step environment, but force reset if timeout exceeded."""
        # Check if episode has been running too long
        elapsed = time.time() - self.episode_start_time
        if elapsed > self.timeout_seconds:
            # Force reset by returning done=True and resetting
            self.forced_resets += 1
            obs, info = self.env.reset()
            # Return as truncated (timeout)
            return obs, 0.0, False, True, {**info, "timeout_forced": True, "timeout_duration": elapsed}
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # If episode ended naturally, reset the timer on next reset
        if terminated or truncated:
            self.episode_start_time = 0.0
        
        return obs, reward, terminated, truncated, info


class EpisodeSuccessLoggerCallback(BaseCallback):
    """Logs per-episode success (goal reached) to CSV during training."""

    def __init__(self, log_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.log_path = Path(log_path)
        self.episode_counter = 0
        self._file = None
        self._writer = None

    def _on_training_start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.log_path.open("w", newline="")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=[
                "episode",
                "timesteps",
                "success",
                "termination_reason",
                "collision_x",
                "collision_y",
                "collision_z",
            ],
        )
        self._writer.writeheader()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        success_flags = []

        for done, info in zip(dones, infos):
            if not done:
                continue

            self.episode_counter += 1
            termination_reason = info.get("termination_reason")
            success = termination_reason == "success"
            success_flags.append(int(success))

            # Extract collision coordinates if available
            collisions = info.get("collisions", [])
            collision_x = ""
            collision_y = ""
            collision_z = ""
            
            if collisions:
                # Use the first collision's contact point
                first_collision = collisions[0]
                if hasattr(first_collision, "contact_point"):
                    contact_point = first_collision.contact_point
                    collision_x = f"{contact_point[0]:.3f}"
                    collision_y = f"{contact_point[1]:.3f}"
                    collision_z = f"{contact_point[2]:.3f}"
                elif isinstance(first_collision, dict):
                    # Handle case where collision is already a dict
                    contact_point = first_collision.get("contact_point", [])
                    if contact_point:
                        collision_x = f"{contact_point[0]:.3f}"
                        collision_y = f"{contact_point[1]:.3f}"
                        collision_z = f"{contact_point[2]:.3f}"

            if self._writer:
                self._writer.writerow(
                    {
                        "episode": self.episode_counter,
                        "timesteps": self.num_timesteps,
                        "success": success,
                        "termination_reason": termination_reason or "",
                        "collision_x": collision_x,
                        "collision_y": collision_y,
                        "collision_z": collision_z,
                    }
                )
                self._file.flush()

        if success_flags:
            # Record fraction of successful episodes finished on this step
            self.logger.record("rollout/ep_success_rate", float(np.mean(success_flags)))

        return True

    def _on_training_end(self) -> None:
        if self._file:
            self._file.close()

TRAINING_CONFIG = {
    # Training parameters
    "total_steps": int(1e6),  # Total training timesteps
    "save_freq": 50000,  # Save checkpoint every N steps
    "eval_freq": 10000,  # Evaluate every N steps
    "eval_episodes": 10,  # Number of episodes per evaluation

    # Environment parameters
    "env_spec": "random_obstacles",  # Environment spec: "empty", "simple", "manual_only", or "random_obstacles"
    "max_steps": 2000,  # Maximum steps per episode (~8.3s at 240Hz, ~33s at 60Hz)
    "drone_radius": 0.5,  # Drone collision radius in meters

    # PPO hyperparameters
    "learning_rate": 1e-3,  # Learning rate (increased to learn faster)
    "n_steps": 2048,  # Steps collected before each policy update
    "batch_size": 64,  # Minibatch size for training
    "n_epochs": 10,  # Number of optimization epochs per update
    "gamma": 0.99,  # Discount factor
    "gae_lambda": 0.95,  # GAE lambda parameter
    "clip_range": 0.2,  # PPO clip range
    "ent_coef": 0.05,  # Entropy coefficient for exploration (increased to escape local optima)

    # Parallelization
    "n_envs": 8,  # Number of parallel environments
    "use_subproc": True,  # Use SubprocVecEnv (True) or DummyVecEnv (False)
    "vec_env_timeout": 30.0,  # Timeout for VecEnv operations (prevents SubprocVecEnv deadlocks)
    "env_timeout_seconds": 60.0,  # Force reset envs stuck longer than this (prevents stuck episodes)

    # Normalization (recommended for stable training)
    "normalize_obs": True,  # Normalize observations
    "normalize_reward": True,  # Normalize rewards

    # Output directories
    "log_dir": "logs",  # Directory for tensorboard logs
    "model_dir": "models",  # Directory for saved models
    "run_name": None,  # Name for this run (None = auto-generate timestamp)

    # Resumption
    "resume": None,  # Path to model.zip to resume training (None = start fresh)
}

def make_env(max_steps: int, drone_radius: float, config: dict = None):
    """
    Create a single FlockRL environment.

    Args:
        max_steps: Maximum steps per episode
        drone_radius: Drone collision radius
        config: Training config dict (for env_spec and timeout settings)
    """
    if config is None:
        config = {}

    def _init():
        # Load environment spec from config
        env_spec_name = config.get("env_spec", "manual_only")
        loader = EnvironmentSpecLoader()
        spec = loader.load(env_spec_name)
        environment = EnvironmentBuilder.from_spec(spec).build()

        # Create gym environment
        env = FlockRLGymEnv(
            environment=environment,
            sim_config=SimulationConfig(
                max_steps=max_steps,
                terminate_on_collision=True,
                goal_threshold=0.5,
            ),
            drone_radius=drone_radius,
            enable_collisions=True,
            # Reward shaping - tuned to encourage goal reaching
            success_reward=200.0,  # Increased to make success very attractive
            collision_penalty=30.0,  # Reduced to allow more exploration
            step_cost=0.02,  # Further reduced to allow longer episodes
            distance_scale=5.0,  # Significantly increased to reward progress toward goal
        )

        # Wrap with Monitor for logging
        env = Monitor(env)
        
        # Wrap with timeout to prevent stuck environments (important for SubprocVecEnv)
        if config.get("env_timeout_seconds", 0) > 0:
            env = TimeoutWrapper(env, timeout_seconds=config["env_timeout_seconds"])
        
        return env

    return _init


def train(config):
    """Main training function."""

    # Create run name if not provided
    if config["run_name"] is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config["run_name"] = f"ppo_simple_{timestamp}"

    print("=" * 80)
    print(f"FlockRL PPO Training: {config['run_name']}")
    print("=" * 80)
    print(f"\nEnvironment: {config['env_spec']}.json")
    print(f"Total timesteps: {config['total_steps']:,}")
    print(f"Parallel environments: {config['n_envs']}")
    print(f"Max steps per episode: {config['max_steps']}")
    print(f"Drone radius: {config['drone_radius']}m")

    # Create directories
    log_dir = Path(config["log_dir"]) / config["run_name"]
    model_dir = Path(config["model_dir"]) / config["run_name"]
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nLog directory: {log_dir}")
    print(f"Model directory: {model_dir}")

    # Create vectorized training environment
    # Choose vec_env_cls based on config
    vec_env_cls = SubprocVecEnv if config.get("use_subproc", False) else DummyVecEnv
    env_type = "multiprocessing" if config.get("use_subproc", False) else "sequential"
    print(f"\nCreating {config['n_envs']} parallel environments ({env_type})...")
    if config.get("env_timeout_seconds", 0) > 0:
        print(f"  Episode timeout protection: {config['env_timeout_seconds']}s per episode")
    if config.get("use_subproc", False) and config.get("vec_env_timeout", 0) > 0:
        print(f"  VecEnv timeout protection: {config['vec_env_timeout']}s for step/reset operations")
    env = make_vec_env(
        lambda: make_env(config["max_steps"], config["drone_radius"], config)(),
        n_envs=config["n_envs"],
        vec_env_cls=vec_env_cls,
    )
    
    # Wrap with timeout protection if using SubprocVecEnv
    if config.get("use_subproc", False) and config.get("vec_env_timeout", 0) > 0:
        env = TimeoutVecEnv(env, timeout_seconds=config["vec_env_timeout"])

    # Apply normalization if requested
    if config["normalize_obs"] or config["normalize_reward"]:
        print("Applying observation/reward normalization...")
        env = VecNormalize(
            env,
            norm_obs=config["normalize_obs"],
            norm_reward=config["normalize_reward"],
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=config["gamma"],
        )

    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = make_vec_env(
        lambda: make_env(config["max_steps"], config["drone_radius"], config)(),
        n_envs=1,
        vec_env_cls=DummyVecEnv,
    )

    if config["normalize_obs"] or config["normalize_reward"]:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=config["normalize_obs"],
            norm_reward=config["normalize_reward"],
            clip_obs=10.0,
            clip_reward=10.0,
            gamma=config["gamma"],
            training=False,  # Don't update normalization stats during eval
        )

    # PPO hyperparameters
    print("\n" + "=" * 80)
    print("PPO Hyperparameters")
    print("=" * 80)
    print(f"Learning rate: {config['learning_rate']}")
    print(f"N steps: {config['n_steps']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"N epochs: {config['n_epochs']}")
    print(f"Gamma: {config['gamma']}")
    print(f"GAE lambda: {config['gae_lambda']}")
    print(f"Clip range: {config['clip_range']}")
    print(f"Entropy coefficient: {config['ent_coef']}")

    # Create or load model
    if config["resume"]:
        print(f"\nResuming training from: {config['resume']}")
        model = PPO.load(config["resume"], env=env)
        if config["normalize_obs"] or config["normalize_reward"]:
            # Load normalization stats
            vec_normalize_path = Path(config["resume"]).parent / "vec_normalize.pkl"
            if vec_normalize_path.exists():
                env = VecNormalize.load(str(vec_normalize_path), env)
                print(f"Loaded normalization stats from: {vec_normalize_path}")
    else:
        print("\nCreating new PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=config["learning_rate"],
            n_steps=config["n_steps"],
            batch_size=config["batch_size"],
            n_epochs=config["n_epochs"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
            clip_range=config["clip_range"],
            ent_coef=config["ent_coef"],
            verbose=1,
        )

    # Set up logger (must be done after model creation)
    # This configures logging to stdout, CSV, and TensorBoard
    new_logger = configure(str(log_dir), ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    # Create callbacks
    callbacks = []

    # Per-episode goal reach logging
    episode_success_logger = EpisodeSuccessLoggerCallback(
        log_path=log_dir / "episode_outcomes.csv"
    )
    callbacks.append(episode_success_logger)

    # Checkpoint callback - save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=config["save_freq"],
        save_path=str(model_dir),
        name_prefix="ppo_flockrl",
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    # Evaluation callback - evaluate and save best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(log_dir / "eval"),
        eval_freq=config["eval_freq"],
        n_eval_episodes=config["eval_episodes"],
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)

    callback_list = CallbackList(callbacks)

    # Train
    print("\n" + "=" * 80)
    print("Starting Training")
    print("=" * 80 + "\n")

    try:
        model.learn(
            total_timesteps=config["total_steps"],
            callback=callback_list,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")

    # Save final model
    final_model_path = model_dir / "final_model"
    print(f"\nSaving final model to: {final_model_path}")
    model.save(str(final_model_path))

    # Save normalization stats if used
    if config["normalize_obs"] or config["normalize_reward"]:
        vec_normalize_path = model_dir / "final_vec_normalize.pkl"
        env.save(str(vec_normalize_path))
        print(f"Saved normalization stats to: {vec_normalize_path}")

    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"\nFinal model: {final_model_path}.zip")
    print(f"Best model: {model_dir / 'best' / 'best_model.zip'}")
    print(f"Logs: {log_dir}")
    print(f"\nTo visualize training progress:")
    print(f"  tensorboard --logdir {log_dir}")

    return model


if __name__ == "__main__":
    train(TRAINING_CONFIG)
