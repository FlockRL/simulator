"""
SB3 PPO Training Script for FlockRL with all hyperparameters exposed.

This script provides a complete training setup with:
- All PPO hyperparameters configurable via command line or config dict
- Proper observation/action space flattening for SB3 compatibility
- Console logging (no TensorBoard required)
- Checkpointing and evaluation callbacks
- Customizable reward function

Usage:
    # Basic training
    python train_sb3_ppo.py

    # With custom hyperparameters
    python train_sb3_ppo.py --total_timesteps 500000 --learning_rate 1e-4 --n_envs 8

    # Resume from checkpoint
    python train_sb3_ppo.py --resume runs/ppo_flockrl_20240101_120000/checkpoints/model_100000_steps.zip
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

# SB3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

# FlockRL imports
from flockrl_sim import (
    FlockRLGymEnv,
    RewardFunction,
    SwarmState,
    load_config,
    load_environment_from_spec,
)


# =============================================================================
# REWARD FUNCTION
# =============================================================================

class TrainingRewardFunction(RewardFunction):
    """
    Configurable reward function for PPO training.

    Reward components:
    - Progress reward: positive for moving toward goal
    - Step penalty: small negative per step to encourage efficiency
    - Success bonus: large positive on goal reached
    - Collision penalty: large negative on collision
    - Action penalty: penalize large accelerations (optional)
    """

    def __init__(
        self,
        progress_scale: float = 20.0,
        step_penalty: float = 0.005,
        success_bonus: float = 10.0,
        collision_penalty: float = 100.0,
        action_penalty: float = 0.0,
        timeout_penalty: float = 5.0,
    ):
        self.progress_scale = progress_scale
        self.step_penalty = step_penalty
        self.success_bonus = success_bonus
        self.collision_penalty = collision_penalty
        self.action_penalty = action_penalty
        self.timeout_penalty = timeout_penalty
        self._last_distances: Optional[np.ndarray] = None

    def reset(self, state: SwarmState) -> None:
        self._last_distances = np.linalg.norm(state.pos - state.goals, axis=1)

    def compute(
        self, state: SwarmState, action: np.ndarray, sim_info: Dict[str, Any]
    ) -> np.ndarray:
        current_distances = np.linalg.norm(state.pos - state.goals, axis=1)

        # Progress reward (positive when moving toward goal)
        progress = (self._last_distances - current_distances) * self.progress_scale

        # Base reward: progress minus step cost
        rewards = progress - self.step_penalty

        # Action penalty (penalize large accelerations)
        if self.action_penalty > 0:
            action_magnitude = np.linalg.norm(action, axis=1)
            rewards -= self.action_penalty * action_magnitude

        # Terminal rewards
        termination_reason = sim_info.get("termination_reason")
        if termination_reason == "success":
            rewards += self.success_bonus
        elif termination_reason == "collision":
            rewards -= self.collision_penalty
        elif termination_reason == "timeout":
            rewards -= self.timeout_penalty

        self._last_distances = current_distances
        return rewards


# =============================================================================
# SB3-COMPATIBLE WRAPPER
# =============================================================================

class FlattenedFlockRLEnv(gym.Wrapper):
    """
    Wrapper that flattens observation and action spaces for SB3 compatibility.

    FlockRLGymEnv uses shape (num_drones, dim) but SB3 expects flat vectors.
    This wrapper handles the reshaping automatically.
    """

    def __init__(self, env: FlockRLGymEnv):
        super().__init__(env)

        # Get original shapes - these are Box spaces with .shape attribute
        obs_space = env.observation_space
        act_space = env.action_space

        assert obs_space.shape is not None, "Observation space must have shape"
        assert act_space.shape is not None, "Action space must have shape"

        self.num_drones = obs_space.shape[0]
        self.obs_dim = obs_space.shape[1]
        self.act_dim = act_space.shape[1]

        # Create flattened spaces
        flat_obs_dim = self.num_drones * self.obs_dim
        flat_act_dim = self.num_drones * self.act_dim

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(flat_obs_dim,),
            dtype=np.float32,
        )

        # Access low/high from the Box action space
        assert isinstance(act_space, spaces.Box), "Action space must be Box"
        self.action_space = spaces.Box(
            low=act_space.low.flatten(),
            high=act_space.high.flatten(),
            shape=(flat_act_dim,),
            dtype=np.float32,
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        return obs.flatten().astype(np.float32), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Reshape flat action to (num_drones, 3)
        action_reshaped = action.reshape(self.num_drones, self.act_dim)
        obs, reward, terminated, truncated, info = self.env.step(action_reshaped)

        # Flatten observation and sum rewards across drones
        flat_obs = obs.flatten().astype(np.float32)
        total_reward = float(np.sum(reward))  # Sum rewards for all drones

        return flat_obs, total_reward, terminated, truncated, info


# =============================================================================
# CONSOLE LOGGING CALLBACK
# =============================================================================

class ConsoleLoggerCallback(BaseCallback):
    """
    Callback that logs training progress to the console.

    Tracks:
    - Episode rewards, lengths
    - Success/collision/timeout rates
    - Policy loss, value loss, entropy
    """

    def __init__(
        self,
        log_freq: int = 5000,
        rolling_window: int = 100,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.rolling_window = rolling_window

        # Episode tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.successes: List[int] = []
        self.collisions: List[int] = []
        self.timeouts: List[int] = []

        # For tracking best performance
        self.best_success_rate = 0.0
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        # Check for episode end in infos
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

            termination_reason = info.get("termination_reason")
            if termination_reason:
                self.successes.append(1 if termination_reason == "success" else 0)
                self.collisions.append(1 if termination_reason == "collision" else 0)
                self.timeouts.append(1 if termination_reason == "timeout" else 0)

        # Log at specified frequency
        if self.n_calls % self.log_freq == 0 and len(self.episode_rewards) > 0:
            self._log_stats()

        return True

    def _log_stats(self) -> None:
        """Print training statistics to console."""
        n = min(self.rolling_window, len(self.episode_rewards))

        # Calculate rolling stats
        mean_reward = np.mean(self.episode_rewards[-n:])
        mean_length = np.mean(self.episode_lengths[-n:])

        success_rate = np.mean(self.successes[-n:]) if self.successes else 0.0
        collision_rate = np.mean(self.collisions[-n:]) if self.collisions else 0.0
        timeout_rate = np.mean(self.timeouts[-n:]) if self.timeouts else 0.0

        # Track best
        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward

        # Build log message
        print("\n" + "=" * 70)
        print(f"Step: {self.num_timesteps:,} | Episodes: {len(self.episode_rewards)}")
        print("-" * 70)
        print(f"  Reward:     mean={mean_reward:8.2f}  (best={self.best_mean_reward:.2f})")
        print(f"  Length:     mean={mean_length:8.1f}")
        print(f"  Success:    {success_rate:6.1%}  (best={self.best_success_rate:.1%})")
        print(f"  Collision:  {collision_rate:6.1%}")
        print(f"  Timeout:    {timeout_rate:6.1%}")
        print("=" * 70)

    def _on_training_end(self) -> None:
        """Print final summary."""
        if len(self.episode_rewards) == 0:
            return

        print("\n" + "=" * 70)
        print("TRAINING COMPLETE")
        print("=" * 70)
        print(f"Total episodes: {len(self.episode_rewards)}")
        print(f"Total timesteps: {self.num_timesteps:,}")
        print("-" * 70)
        print(f"Final mean reward: {np.mean(self.episode_rewards[-100:]):.2f}")
        print(f"Best mean reward: {self.best_mean_reward:.2f}")
        print(f"Final success rate: {np.mean(self.successes[-100:]):.1%}")
        print(f"Best success rate: {self.best_success_rate:.1%}")
        print("=" * 70)


# =============================================================================
# ENVIRONMENT FACTORY
# =============================================================================

def make_env(
    env_spec: str,
    reward_config: Dict[str, float],
    config_path: Optional[Path] = None,
    rank: int = 0,
    seed: int = 0,
) -> Callable[[], gym.Env]:
    """
    Factory function to create environments for vectorized training.
    """
    def _init() -> gym.Env:
        config = load_config(config_path)
        environment = load_environment_from_spec(env_spec, config)

        reward_fn = TrainingRewardFunction(**reward_config)

        env = FlockRLGymEnv(
            reward_fn=reward_fn,
            environment=environment,
            config_path=config_path,
        )

        # Wrap for SB3 compatibility
        env = FlattenedFlockRLEnv(env)
        env = Monitor(env)

        # Set seed
        env.reset(seed=seed + rank)

        return env

    set_random_seed(seed)
    return _init


# =============================================================================
# HYPERPARAMETER CONFIG
# =============================================================================

def get_default_hyperparams() -> Dict[str, Any]:
    """
    Returns all PPO hyperparameters with sensible defaults for FlockRL.

    These are the main knobs to tune for your problem.
    """
    return {
        # === CORE PPO HYPERPARAMETERS ===

        # Learning rate - start here for tuning
        # Lower (1e-4 to 3e-4) for stability, higher (3e-4 to 1e-3) for faster learning
        "learning_rate": 3e-4,

        # Number of steps to collect before each policy update
        # Higher = more stable gradients, but slower updates
        # Should be divisible by batch_size. Common: 2048, 4096, 8192
        "n_steps": 1024,

        # Mini-batch size for gradient updates
        # Smaller = noisier gradients, larger = smoother but more memory
        # Must divide n_steps * n_envs evenly
        "batch_size": 256,

        # Number of epochs to train on collected rollout data
        # Higher = more sample efficient, but risk of overfitting
        # Common: 4-10
        "n_epochs": 5,

        # Discount factor (gamma) - how much to value future rewards
        # Higher (0.99) for long-horizon tasks, lower (0.9-0.95) for short
        "gamma": 0.99,

        # GAE lambda - bias-variance tradeoff for advantage estimation
        # Higher (0.95-0.99) = lower bias, higher variance
        # Lower (0.9-0.95) = higher bias, lower variance
        "gae_lambda": 0.95,

        # PPO clipping parameter - limits policy update magnitude
        # Lower (0.1) for stability, higher (0.3) for faster adaptation
        "clip_range": 0.2,

        # Value function clipping (None to disable)
        # Set same as clip_range, or None
        "clip_range_vf": None,

        # Entropy coefficient - encourages exploration
        # WARNING: For continuous actions, high values cause std explosion!
        # Use 0.0 or very small (0.001) for continuous control
        # For continuous control without SDE, set to 0 - Gaussian std handles exploration
        "ent_coef": 0.0,

        # Value function coefficient in loss
        # Usually 0.5, can increase if value function learning is slow
        "vf_coef": 0.5,

        # Max gradient norm for clipping
        # Prevents exploding gradients. Common: 0.5-1.0
        "max_grad_norm": 0.5,

        # === NETWORK ARCHITECTURE ===

        # Policy network architecture
        # "MlpPolicy" uses the net_arch below
        "policy": "MlpPolicy",

        # Network architecture: [hidden_layer_sizes]
        # Shared layers first, then separate pi (policy) and vf (value) heads
        # Example: dict(pi=[256, 256], vf=[256, 256]) for separate networks
        "net_arch": [128, 128],

        # Activation function: "tanh", "relu", "elu", "leaky_relu"
        "activation_fn": "tanh",

        # Initial log standard deviation for action noise
        # Default 0 (std=1.0) is way too noisy - actions are random
        # -1.0 gives std=0.37: coherent enough to develop directed movement
        "log_std_init": -1.0,

        # === ADVANCED OPTIONS ===

        # Whether to use generalized State Dependent Exploration
        # SDE can help in some continuous control tasks but prevents convergence here
        "use_sde": False,

        # Sample new noise matrix every n steps when using SDE (irrelevant when SDE off)
        "sde_sample_freq": -1,

        # Normalize advantage estimates (recommended True)
        "normalize_advantage": True,

        # Target KL divergence for early stopping (None to disable)
        # Critical for preventing destructive updates - stops epoch early if KL exceeds this
        "target_kl": 0.03,

        # === TRAINING SETTINGS ===

        # Total training timesteps
        "total_timesteps": 800_000,

        # Number of parallel environments
        "n_envs": 8,

        # Use subprocess vectorized envs (faster but more memory)
        "use_subproc": True,

        # Whether to normalize observations
        # WARNING: Can distort goal vector/distance - try False first!
        "normalize_obs": False,

        # Whether to normalize rewards
        "normalize_reward": True,

        # === REWARD FUNCTION ===

        "reward_config": {
            "progress_scale": 20.0,
            "step_penalty": 0.005,
            "success_bonus": 10.0,
            "collision_penalty": 100.0,
            "action_penalty": 0.0,
            "timeout_penalty": 5.0,
        },

        # === ENVIRONMENT ===
        # "empty" = no obstacles, "simple" = walls with 2m gates
        "env_spec": "empty",

        # === LOGGING & CHECKPOINTING ===

        "checkpoint_freq": 10_000_000,  # Save model every N steps (set higher than total to skip)
        "eval_freq": 20_000,  # Evaluate every N steps
        "n_eval_episodes": 10,  # Episodes per evaluation
        "log_freq": 10_000,  # Console log frequency
        "verbose": 1,  # 0=no output, 1=info, 2=debug
    }


# =============================================================================
# TRAINING FUNCTION
# =============================================================================

def train(
    hyperparams: Dict[str, Any],
    run_name: Optional[str] = None,
    resume_path: Optional[str] = None,
    config_path: Optional[Path] = None,
    seed: int = 42,
) -> PPO:
    """
    Train a PPO agent with the given hyperparameters.

    Args:
        hyperparams: Dictionary of hyperparameters (see get_default_hyperparams())
        run_name: Name for this training run (auto-generated if None)
        resume_path: Path to model checkpoint to resume from
        config_path: Path to FlockRL config.yml
        seed: Random seed for reproducibility

    Returns:
        Trained PPO model
    """
    # Create run directory
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"ppo_flockrl_{timestamp}"

    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = run_dir / "checkpoints"
    eval_dir = run_dir / "eval"

    checkpoint_dir.mkdir(exist_ok=True)
    eval_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("FlockRL PPO Training")
    print("=" * 70)
    print(f"Run directory: {run_dir}")
    print(f"Seed: {seed}")
    print()

    # Save hyperparameters
    with open(run_dir / "hyperparams.json", "w") as f:
        hp_save = hyperparams.copy()
        hp_save["activation_fn"] = str(hp_save["activation_fn"])
        json.dump(hp_save, f, indent=2)

    # Create vectorized training environment
    n_envs = hyperparams["n_envs"]
    env_fns = [
        make_env(
            env_spec=hyperparams["env_spec"],
            reward_config=hyperparams["reward_config"],
            config_path=config_path,
            rank=i,
            seed=seed,
        )
        for i in range(n_envs)
    ]

    if hyperparams["use_subproc"] and n_envs > 1:
        vec_env = SubprocVecEnv(env_fns)
    else:
        vec_env = DummyVecEnv(env_fns)

    # Optionally normalize observations and rewards
    if hyperparams["normalize_obs"] or hyperparams["normalize_reward"]:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=hyperparams["normalize_obs"],
            norm_reward=hyperparams["normalize_reward"],
            clip_obs=10.0,
            clip_reward=10.0,
        )

    # Create evaluation environment
    eval_env_fn = make_env(
        env_spec=hyperparams["env_spec"],
        reward_config=hyperparams["reward_config"],
        config_path=config_path,
        rank=0,
        seed=seed + 1000,
    )
    eval_env = DummyVecEnv([eval_env_fn])
    if hyperparams["normalize_obs"] or hyperparams["normalize_reward"]:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=hyperparams["normalize_obs"],
            norm_reward=False,  # Don't normalize eval rewards for true performance
            clip_obs=10.0,
            training=False,  # Don't update stats during eval
        )

    # Map activation function string to torch class
    activation_name = hyperparams["activation_fn"]
    activation_classes = {
        "tanh": torch.nn.Tanh,
        "relu": torch.nn.ReLU,
        "elu": torch.nn.ELU,
        "leaky_relu": torch.nn.LeakyReLU,
    }
    activation_fn = activation_classes.get(activation_name, torch.nn.Tanh)

    # Policy kwargs
    policy_kwargs = {
        "net_arch": hyperparams["net_arch"],
        "activation_fn": activation_fn,
        "log_std_init": hyperparams["log_std_init"],
    }

    # Create or load model (no tensorboard_log - we use console logging)
    if resume_path:
        print(f"Resuming from: {resume_path}")
        model = PPO.load(
            resume_path,
            env=vec_env,
        )
        # Load VecNormalize stats if they exist
        vec_norm_path = Path(resume_path).parent / "vec_normalize.pkl"
        if vec_norm_path.exists() and isinstance(vec_env, VecNormalize):
            vec_env = VecNormalize.load(str(vec_norm_path), vec_env)
    else:
        model = PPO(
            policy=hyperparams["policy"],
            env=vec_env,
            learning_rate=hyperparams["learning_rate"],
            n_steps=hyperparams["n_steps"],
            batch_size=hyperparams["batch_size"],
            n_epochs=hyperparams["n_epochs"],
            gamma=hyperparams["gamma"],
            gae_lambda=hyperparams["gae_lambda"],
            clip_range=hyperparams["clip_range"],
            clip_range_vf=hyperparams["clip_range_vf"],
            ent_coef=hyperparams["ent_coef"],
            vf_coef=hyperparams["vf_coef"],
            max_grad_norm=hyperparams["max_grad_norm"],
            use_sde=hyperparams["use_sde"],
            sde_sample_freq=hyperparams["sde_sample_freq"],
            normalize_advantage=hyperparams["normalize_advantage"],
            target_kl=hyperparams["target_kl"],
            policy_kwargs=policy_kwargs,
            verbose=hyperparams["verbose"],
            seed=seed,
        )

    # Print model summary
    print("Model Architecture:")
    print(f"  Policy: {hyperparams['policy']}")
    print(f"  Net arch: {hyperparams['net_arch']}")
    print(f"  Activation: {hyperparams['activation_fn']}")
    print()
    print("Key Hyperparameters:")
    print(f"  Learning rate: {hyperparams['learning_rate']}")
    print(f"  Batch size: {hyperparams['batch_size']}")
    print(f"  N steps: {hyperparams['n_steps']}")
    print(f"  N epochs: {hyperparams['n_epochs']}")
    print(f"  Gamma: {hyperparams['gamma']}")
    print(f"  GAE lambda: {hyperparams['gae_lambda']}")
    print(f"  Clip range: {hyperparams['clip_range']}")
    print(f"  Entropy coef: {hyperparams['ent_coef']}")
    print()
    print("Training:")
    print(f"  Total timesteps: {hyperparams['total_timesteps']:,}")
    print(f"  Parallel envs: {n_envs}")
    print(f"  Normalize obs: {hyperparams['normalize_obs']}")
    print(f"  Normalize reward: {hyperparams['normalize_reward']}")
    print("=" * 70)
    print()

    # Setup callbacks
    callbacks = []

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(hyperparams["checkpoint_freq"] // n_envs, 1),
        save_path=str(checkpoint_dir),
        name_prefix="model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    callbacks.append(checkpoint_callback)

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(eval_dir),
        eval_freq=max(hyperparams["eval_freq"] // n_envs, 1),
        n_eval_episodes=hyperparams["n_eval_episodes"],
        deterministic=True,
        render=False,
        verbose=0,  # Quiet - we have our own logging
    )
    callbacks.append(eval_callback)

    # Console logging callback
    console_callback = ConsoleLoggerCallback(
        log_freq=hyperparams["log_freq"],
        rolling_window=100,
    )
    callbacks.append(console_callback)

    callback = CallbackList(callbacks)

    # Train!
    try:
        model.learn(
            total_timesteps=hyperparams["total_timesteps"],
            callback=callback,
            reset_num_timesteps=not bool(resume_path),
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")

    # Save final model
    final_model_path = run_dir / "final_model"
    model.save(str(final_model_path))
    if isinstance(vec_env, VecNormalize):
        vec_env.save(str(run_dir / "vec_normalize.pkl"))

    print()
    print("Training complete!")
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved to: {run_dir / 'best_model'}")

    # Cleanup
    vec_env.close()
    eval_env.close()

    return model


# =============================================================================
# MAIN
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO on FlockRL environment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Get defaults
    defaults = get_default_hyperparams()

    # Core hyperparameters
    parser.add_argument("--learning_rate", type=float, default=defaults["learning_rate"],
                        help="Learning rate")
    parser.add_argument("--n_steps", type=int, default=defaults["n_steps"],
                        help="Steps per rollout")
    parser.add_argument("--batch_size", type=int, default=defaults["batch_size"],
                        help="Mini-batch size")
    parser.add_argument("--n_epochs", type=int, default=defaults["n_epochs"],
                        help="Epochs per update")
    parser.add_argument("--gamma", type=float, default=defaults["gamma"],
                        help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=defaults["gae_lambda"],
                        help="GAE lambda")
    parser.add_argument("--clip_range", type=float, default=defaults["clip_range"],
                        help="PPO clip range")
    parser.add_argument("--ent_coef", type=float, default=defaults["ent_coef"],
                        help="Entropy coefficient")
    parser.add_argument("--vf_coef", type=float, default=defaults["vf_coef"],
                        help="Value function coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=defaults["max_grad_norm"],
                        help="Max gradient norm")

    # Network architecture
    parser.add_argument("--net_arch", type=int, nargs="+", default=defaults["net_arch"],
                        help="Hidden layer sizes")
    parser.add_argument("--activation", type=str, default=defaults["activation_fn"],
                        choices=["tanh", "relu", "elu", "leaky_relu"],
                        help="Activation function")

    # Training settings
    parser.add_argument("--total_timesteps", type=int, default=defaults["total_timesteps"],
                        help="Total training timesteps")
    parser.add_argument("--n_envs", type=int, default=defaults["n_envs"],
                        help="Number of parallel environments")
    parser.add_argument("--no_subproc", action="store_true",
                        help="Disable subprocess vectorized envs")
    parser.add_argument("--no_normalize_obs", action="store_true",
                        help="Disable observation normalization")
    parser.add_argument("--no_normalize_reward", action="store_true",
                        help="Disable reward normalization")

    # Reward function
    parser.add_argument("--progress_scale", type=float,
                        default=defaults["reward_config"]["progress_scale"],
                        help="Progress reward scale")
    parser.add_argument("--step_penalty", type=float,
                        default=defaults["reward_config"]["step_penalty"],
                        help="Per-step penalty")
    parser.add_argument("--success_bonus", type=float,
                        default=defaults["reward_config"]["success_bonus"],
                        help="Success bonus")
    parser.add_argument("--collision_penalty", type=float,
                        default=defaults["reward_config"]["collision_penalty"],
                        help="Collision penalty")
    parser.add_argument("--action_penalty", type=float,
                        default=defaults["reward_config"]["action_penalty"],
                        help="Action magnitude penalty")
    parser.add_argument("--timeout_penalty", type=float,
                        default=defaults["reward_config"]["timeout_penalty"],
                        help="Timeout penalty")

    # Environment
    parser.add_argument("--env_spec", type=str, default=defaults["env_spec"],
                        help="Environment spec name or path")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Path to FlockRL config.yml")

    # Logging
    parser.add_argument("--checkpoint_freq", type=int, default=defaults["checkpoint_freq"],
                        help="Checkpoint frequency (steps)")
    parser.add_argument("--eval_freq", type=int, default=defaults["eval_freq"],
                        help="Evaluation frequency (steps)")
    parser.add_argument("--log_freq", type=int, default=defaults["log_freq"],
                        help="Console log frequency (steps)")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name for this run")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--verbose", type=int, default=defaults["verbose"],
                        help="Verbosity level (0-2)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Build hyperparams dict from args
    hyperparams = get_default_hyperparams()

    # Override with command line args
    hyperparams.update({
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "net_arch": args.net_arch,
        "activation_fn": args.activation,
        "total_timesteps": args.total_timesteps,
        "n_envs": args.n_envs,
        "use_subproc": not args.no_subproc,
        "normalize_obs": not args.no_normalize_obs,
        "normalize_reward": not args.no_normalize_reward,
        "env_spec": args.env_spec,
        "checkpoint_freq": args.checkpoint_freq,
        "eval_freq": args.eval_freq,
        "log_freq": args.log_freq,
        "verbose": args.verbose,
        "reward_config": {
            "progress_scale": args.progress_scale,
            "step_penalty": args.step_penalty,
            "success_bonus": args.success_bonus,
            "collision_penalty": args.collision_penalty,
            "action_penalty": args.action_penalty,
            "timeout_penalty": args.timeout_penalty,
        },
    })

    config_path = Path(args.config_path) if args.config_path else None

    train(
        hyperparams=hyperparams,
        run_name=args.run_name,
        resume_path=args.resume,
        config_path=config_path,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
