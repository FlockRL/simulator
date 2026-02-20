"""
Simple PPO training example using Stable-Baselines3.

This script demonstrates:
1. Creating a FlockRL Gym environment with logging enabled
2. Using custom reward functions
3. Training with PPO from Stable-Baselines3
4. Tracking episode outcomes and statistics
5. Exporting results for analysis
"""

import numpy as np
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Tuple
import gymnasium as gym
from gymnasium import spaces
from flockrl_sim import FlockRLGymEnv, RewardFunction, SwarmState, load_environment_from_spec, load_config
from flockrl_sim.gym_logging import EpisodeLogger
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

class SimpleRewardFunction(RewardFunction):
    """
    Simplified reward function with strong progress signal.
    
    Key insight: Agent needs POSITIVE reward for making progress, not just
    negative reward for being far away. This provides clearer learning signal.
    
    Reward structure:
    - Progress reward: positive reward for getting closer (scaled by distance improvement)
    - Small step penalty: encourages efficiency
    - Success bonus: large reward when reaching goal
    - Collision penalties: separate penalties for wall vs obstacle collisions
    """
    def __init__(
        self,
        progress_scale: float = 20.0,  # Increased: Reward per unit of distance improvement
        step_penalty: float = 0.05,  # Reduced: Small penalty per step
        success_reward: float = 200.0,  # Large bonus when reaching goal
        collision_penalty: float = -50.0,  # Reduced: Penalty for collisions (used if no specific type)
        wall_collision_penalty: float = -50.0,  # Reduced: Penalty for wall collisions
        obstacle_collision_penalty: float = -50.0,  # Reduced: Penalty for obstacle/clutter collisions
        alive_bonus: float = 0.1,  # Small bonus for staying alive (encourages longer episodes)
    ):
        self.progress_scale = progress_scale
        self.step_penalty = step_penalty
        self.success_reward = success_reward
        self.collision_penalty = collision_penalty
        self.wall_collision_penalty = wall_collision_penalty
        self.obstacle_collision_penalty = obstacle_collision_penalty
        self.alive_bonus = alive_bonus
        self._last_dist = None

    def reset(self, state: SwarmState) -> None:
        # Track initial distance to compute progress
        self._last_dist = np.linalg.norm(state.pos - state.goals, axis=1)

    def compute(self, state: SwarmState, action: np.ndarray, sim_info: Dict[str, Any]) -> np.ndarray:
        # Compute current distance
        curr_dist = np.linalg.norm(state.pos - state.goals, axis=1)
        
        # Progress reward: positive reward for getting closer
        # This is the key - agent gets positive feedback for moving in right direction
        progress = self._last_dist - curr_dist  # Positive when getting closer
        rewards = self.progress_scale * progress - self.step_penalty + self.alive_bonus
        
        # Add termination rewards/penalties
        if sim_info["termination_reason"] == "success":
            rewards += self.success_reward
        elif sim_info["termination_reason"] == "collision":
            # Differentiate between wall and obstacle collisions
            collisions = sim_info.get("collisions", [])
            wall_collision = any(c.collision_type == "wall" for c in collisions)
            obstacle_collision = any(c.collision_type in ("clutter", "sphere") for c in collisions)
            
            if wall_collision:
                rewards += self.wall_collision_penalty
            elif obstacle_collision:
                rewards += self.obstacle_collision_penalty
            else:
                # Fallback for other collision types (bounds, drone, etc.)
                rewards += self.collision_penalty
        
        # Update for next step
        self._last_dist = curr_dist
        
        return rewards


class SingleDroneWrapper(gym.Wrapper):
    """
    Wrapper to extract single-drone observations and actions from multi-drone environment.
    
    PPO/SB3 works with 1D arrays:
    - Actions: PPO outputs (n,) shape (e.g., (3,) for 3D acceleration)
    - Rewards: PPO expects scalar rewards, not arrays
    
    Base FlockRLGymEnv works with 2D arrays (num_drones dimension):
    - Actions: expects (num_drones, 3) shape
    - Observations: returns (num_drones, obs_dim) shape
    - Rewards: returns (num_drones,) array
    
    This wrapper bridges the gap:
    - Observations: (num_drones, obs_dim) → (obs_dim,) [extract first drone]
    - Actions: (3,) → (1, 3) [reshape for base env, since num_drones=1]
    - Rewards: (num_drones,) → scalar [extract first drone's reward]
    """
    
    def __init__(self, env: FlockRLGymEnv):
        super().__init__(env)
        if env.num_drones != 1:
            raise ValueError(
                f"SingleDroneWrapper requires num_drones=1, got {env.num_drones}."
            )
        self.num_drones = env.num_drones
        self.obs_dim = env.observation_space.shape[1]
        self.action_dim = env.action_space.shape[1]
        
        # Extract first drone's observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )
        
        # Extract first drone's action space
        self.action_space = spaces.Box(
            low=env.action_space.low[0],
            high=env.action_space.high[0],
            shape=(self.action_dim,),
            dtype=np.float32,
        )
    
    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        # Extract first drone's observation
        return obs[0], info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool, bool, Dict[str, Any]]:
        # Reshape action from (3,) to (1, 3) for the base environment
        # Base env expects (num_drones, 3) shape, and with num_drones=1, that's (1, 3)
        action_reshaped = action.reshape(1, self.action_dim)
        obs, rewards, terminated, truncated, info = self.env.step(action_reshaped)
        
        # Extract first drone's observation and reward
        obs_single = obs[0]
        reward_single = float(rewards[0])
        
        return obs_single, reward_single, terminated, truncated, info


class TerminationStatsCallback(BaseCallback):
    def __init__(self, log_every_episodes: int = 50, print_after_rollout: bool = True):
        super().__init__()
        self.log_every_episodes = log_every_episodes
        self.print_after_rollout = print_after_rollout
        self.episode_counts = Counter()
        self._episodes = 0
        self._rollout_episodes = Counter()  # Track episodes in current rollout

    @staticmethod
    def _reason_to_tag(reason: str) -> str:
        """Convert termination reason into a TensorBoard-friendly tag."""
        return reason.replace("/", "_").replace(" ", "_")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        
        # Handle vectorized environments: dones is an array, infos is a list
        # Check if empty using safe methods that work with numpy arrays
        try:
            if not infos or (hasattr(infos, '__len__') and len(infos) == 0):
                return True
            if dones is None:
                return True
            # For numpy arrays, use size attribute; for lists, use len
            dones_size = dones.size if hasattr(dones, 'size') else len(dones) if hasattr(dones, '__len__') else 1
            if dones_size == 0:
                return True
        except (TypeError, AttributeError):
            return True
        
        # Convert dones to list if it's a numpy array or other iterable
        try:
            if isinstance(dones, np.ndarray):
                dones = dones.tolist()
            elif hasattr(dones, '__iter__') and not isinstance(dones, (str, dict)):
                dones = list(dones)
            else:
                dones = [dones]
        except (TypeError, ValueError):
            return True
        
        # Ensure infos is a list of per-env info dicts
        # SubprocVecEnv returns a tuple, DummyVecEnv returns a list.
        try:
            if isinstance(infos, tuple):
                infos = list(infos)
            elif isinstance(infos, list):
                pass
            elif hasattr(infos, '__iter__') and not isinstance(infos, (str, dict)):
                infos = list(infos)
            else:
                infos = [infos]
        except (TypeError, ValueError):
            return True
        
        for i, (done, info) in enumerate(zip(dones, infos)):
            if done:
                # Handle info dict (for vectorized envs, info might be a dict or nested)
                # Sometimes info is a list of dicts, sometimes it's a single dict
                if isinstance(info, list) and len(info) > 0:
                    info = info[0]  # Take first element if it's a list
                
                if not isinstance(info, dict):
                    # Try to extract from nested structure
                    if hasattr(info, 'get'):
                        pass  # It might be a dict-like object
                    else:
                        continue
                    
                reason = info.get("termination_reason") if hasattr(info, 'get') else None
                if reason is None:
                    # Try alternative keys or skip
                    continue
                
                # Differentiate between wall and obstacle collisions
                if reason == "collision":
                    collisions = info.get("collisions", [])
                    if isinstance(collisions, list) and len(collisions) > 0:
                        # Check if collisions is a list of objects with collision_type
                        if hasattr(collisions[0], 'collision_type'):
                            wall_collision = any(c.collision_type == "wall" for c in collisions)
                            obstacle_collision = any(c.collision_type in ("clutter", "sphere") for c in collisions)
                        elif isinstance(collisions[0], dict):
                            wall_collision = any(c.get("collision_type") == "wall" for c in collisions)
                            obstacle_collision = any(c.get("collision_type") in ("clutter", "sphere") for c in collisions)
                        else:
                            wall_collision = False
                            obstacle_collision = False
                    else:
                        wall_collision = False
                        obstacle_collision = False
                    
                    if wall_collision:
                        reason = "collision_wall"
                    elif obstacle_collision:
                        reason = "collision_obstacle"
                    # else keep as "collision" for other types (bounds, drone, etc.)
                
                self.episode_counts[reason] += 1
                self._rollout_episodes[reason] += 1
                self._episodes += 1
                if self._episodes % self.log_every_episodes == 0:
                    print(
                        f"Termination stats @ {self._episodes} eps: {dict(self.episode_counts)}"
                    )
        return True
    
    def on_rollout_end(self) -> None:
        """Called after each rollout to print and log termination stats."""
        total_rollout_eps = sum(self._rollout_episodes.values())

        # TensorBoard logging (through SB3 logger)
        self.logger.record("termination/episodes_total", float(self._episodes))
        self.logger.record("termination/rollout_episodes", float(total_rollout_eps))

        if total_rollout_eps > 0:
            for reason, count in self._rollout_episodes.items():
                tag = self._reason_to_tag(reason)
                self.logger.record(f"termination/rollout_count/{tag}", float(count))
                self.logger.record(f"termination/rollout_rate/{tag}", float(count) / float(total_rollout_eps))

        if self._episodes > 0:
            for reason, count in self.episode_counts.items():
                tag = self._reason_to_tag(reason)
                self.logger.record(f"termination/cumulative_count/{tag}", float(count))
                self.logger.record(f"termination/cumulative_rate/{tag}", float(count) / float(self._episodes))

        if self.print_after_rollout and total_rollout_eps > 0:
            print(f"\n  Rollout termination: {dict(self._rollout_episodes)} (total: {total_rollout_eps} eps)\n")

        self._rollout_episodes.clear()  # Reset for next rollout


def run_straight_line_eval(env: gym.Env, episodes: int = 5) -> None:
    """Test if the environment is solvable with a simple straight-line controller."""
    max_accel = float(np.max(env.action_space.high))
    max_steps = int(env.unwrapped.sim_config["max_steps"])
    results = Counter()
    collision_details = Counter()

    print(f"\n{'='*60}")
    print(f"Feasibility Check: Straight-line controller ({episodes} episodes)")
    print(f"{'='*60}")
    
    for _ in range(episodes):
        obs, info = env.reset()
        done = False
        steps = 0
        while not done and steps < max_steps:
            goal_vec = obs[3:6]
            norm = float(np.linalg.norm(goal_vec))
            if norm > 1e-6:
                action = goal_vec / norm * max_accel
            else:
                action = np.zeros(3, dtype=np.float32)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        reason = info.get("termination_reason", "unknown")
        results[reason] += 1
        
        # Track collision types for detailed analysis
        if reason == "collision":
            collisions = info.get("collisions", [])
            wall_collision = any(c.collision_type == "wall" for c in collisions)
            obstacle_collision = any(c.collision_type in ("clutter", "sphere") for c in collisions)
            if wall_collision:
                collision_details["wall"] += 1
            elif obstacle_collision:
                collision_details["obstacle"] += 1
            else:
                collision_details["other"] += 1

    print(f"\nResults: {dict(results)}")
    if collision_details:
        print(f"Collision breakdown: {dict(collision_details)}")
    
    success_rate = results.get("success", 0) / episodes
    print(f"\nSuccess rate: {success_rate:.1%} ({results.get('success', 0)}/{episodes})")
    
    print(f"{'='*60}\n")


def make_env(rank: int, config_path: Path, log_dir: Path, enable_logging: bool = False) -> gym.Env:
    """Create a single environment instance for parallel training."""
    def _init():
        config = load_config(config_path)
        reward_fn = SimpleRewardFunction()
        environment = load_environment_from_spec("large_obstacles_only", config)
        
        flockrl_env = FlockRLGymEnv(
            reward_fn=reward_fn,
            environment=environment,
            config_path=config_path,
        )
        
        env = SingleDroneWrapper(flockrl_env)
        
        # Only enable logging for the first environment to avoid duplicate logs
        if enable_logging and rank == 0:
            flockrl_env.logger = EpisodeLogger(log_dir=log_dir)
        
        # Monitor environment for statistics (each gets its own monitor)
        env = Monitor(
            env,
            str(log_dir / f"monitor_{rank}"),
            info_keywords=("termination_reason",),
        )
        return env
    return _init


def main():
    # Load configuration
    config_path = Path(__file__).parent.parent / "config.yml"
    config = load_config(config_path)
    
    # Create log directory
    log_dir = Path("logs/ppo_training")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Number of parallel environments (use 4-8 cores for MacBook)
    num_envs = 8
    
    # Create vectorized environment with parallel workers
    print(f"\nCreating {num_envs} parallel environments...")
    env = SubprocVecEnv([make_env(i, config_path, log_dir, enable_logging=(i == 0)) for i in range(num_envs)])

    # Create PPO model
    # Fixed hyperparameters to prevent policy std explosion
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=1e-4,  # Much lower learning rate for stability
        n_steps=2048,
        batch_size=64,
        n_epochs=4,  # Reduced epochs to prevent overfitting to bad data
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,  # Tighter clipping to prevent large policy changes
        ent_coef=0.0,  # Remove entropy bonus - let it learn deterministically first
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"log_std_init": -1.0, "ortho_init": False},  # Start with smaller std
        verbose=1,
        tensorboard_log=str(log_dir / "tensorboard"),
    )

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=str(log_dir / "checkpoints"),
        name_prefix="ppo_model",
    )
    termination_callback = TerminationStatsCallback(log_every_episodes=50)
    
    # Train the model
    total_timesteps = 20000000  # Increased from 500k to 20M timesteps
    print(f"Training PPO for {total_timesteps} timesteps...")
    print("=" * 60)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, termination_callback],
        progress_bar=True,
    )

    # Save final model
    model_path = log_dir / "ppo_final_model"
    model.save(str(model_path))
    print(f"\nModel saved to: {model_path}")

    # Save final logs (access first environment from vectorized env)
    # Note: Only the first environment has logging enabled
    try:
        first_env = env.envs[0]
        if hasattr(first_env, 'unwrapped') and hasattr(first_env.unwrapped, 'env'):
            # Unwrap through SingleDroneWrapper to get FlockRLGymEnv
            flockrl_env = first_env.unwrapped.env
            if hasattr(flockrl_env, 'save_episode_logs'):
                flockrl_env.save_episode_logs()
    except Exception as e:
        print(f"Note: Could not save episode logs: {e}")

    print("Training complete!")


if __name__ == "__main__":
    main()
