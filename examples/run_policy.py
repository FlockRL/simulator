"""
Run a trained RL policy on an environment and optionally visualize the flight path.

This script:
1. Loads a trained PPO model from Stable-Baselines3
2. Runs it on a specified environment
3. Saves the simulation run for visualization
4. Optionally opens the visualization

Usage:
    python examples/run_policy.py <model_path> [options]
    
Examples:
    # Run policy and save logs
    python examples/run_policy.py logs/ppo_training/ppo_final_model.zip
    
    # Run on specific environment
    python examples/run_policy.py logs/ppo_training/ppo_final_model.zip --env rand_large_obstacles_only
    
    # Run and automatically visualize
    python examples/run_policy.py logs/ppo_training/ppo_final_model.zip --visualize
    
    # Run multiple episodes
    python examples/run_policy.py logs/ppo_training/ppo_final_model.zip --episodes 5
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

from flockrl_sim import FlockRLGymEnv, RewardFunction, SwarmState, load_environment_from_spec, load_config
from flockrl_sim.visualization import OfflineVisualizer


class SimpleRewardFunction(RewardFunction):
    """Reward function matching the training setup."""
    def __init__(
        self,
        progress_scale: float = 20.0,
        step_penalty: float = 0.05,
        success_reward: float = 200.0,
        collision_penalty: float = -50.0,
        wall_collision_penalty: float = -50.0,
        obstacle_collision_penalty: float = -50.0,
        alive_bonus: float = 0.1,
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
        self._last_dist = np.linalg.norm(state.pos - state.goals, axis=1)

    def compute(self, state: SwarmState, action: np.ndarray, sim_info: dict) -> np.ndarray:
        curr_dist = np.linalg.norm(state.pos - state.goals, axis=1)
        progress = self._last_dist - curr_dist
        rewards = self.progress_scale * progress - self.step_penalty + self.alive_bonus
        
        if sim_info["termination_reason"] == "success":
            rewards += self.success_reward
        elif sim_info["termination_reason"] == "collision":
            collisions = sim_info.get("collisions", [])
            wall_collision = any(c.collision_type == "wall" for c in collisions)
            obstacle_collision = any(c.collision_type in ("clutter", "sphere") for c in collisions)
            
            if wall_collision:
                rewards += self.wall_collision_penalty
            elif obstacle_collision:
                rewards += self.obstacle_collision_penalty
            else:
                rewards += self.collision_penalty
        
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
        from gymnasium import spaces
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
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs[0], info
    
    def step(self, action: np.ndarray):
        action_reshaped = action.reshape(1, self.action_dim)
        obs, rewards, terminated, truncated, info = self.env.step(action_reshaped)
        obs_single = obs[0]
        reward_single = float(rewards[0])
        return obs_single, reward_single, terminated, truncated, info


def run_policy(
    model_path: Path,
    env_spec: Optional[str] = None,
    config_path: Optional[Path] = None,
    episodes: int = 1,
    seed: Optional[int] = None,
    save_runs: bool = True,
) -> list[Path]:
    """
    Run a trained policy on an environment.
    
    Args:
        model_path: Path to the trained PPO model (.zip file)
        env_spec: Environment spec name (default: from config or "large_obstacles_only")
        config_path: Path to config file (default: config.yml in project root)
        episodes: Number of episodes to run
        seed: Random seed for environment
        save_runs: Whether to save simulation runs for visualization
        
    Returns:
        List of paths to saved episode log files
    """
    # Load model
    print(f"Loading model from: {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = PPO.load(str(model_path))
    print(f"✓ Model loaded successfully")
    
    # Load config
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config.yml"
    config = load_config(config_path)
    
    # Override save_runs setting
    if save_runs:
        config["gym"]["save_runs"] = True
        if not config["gym"].get("log_dir"):
            # Create a default log directory if not set
            log_dir = Path("logs/policy_runs")
            log_dir.mkdir(parents=True, exist_ok=True)
            config["gym"]["log_dir"] = str(log_dir)
    
    # Determine environment spec
    if env_spec is None:
        env_spec = config.get("environment", {}).get("spec") or "large_obstacles_only"
    
    print(f"Environment spec: {env_spec}")
    
    # Create environment
    reward_fn = SimpleRewardFunction()
    environment = load_environment_from_spec(env_spec, config)
    
    env = FlockRLGymEnv(
        reward_fn=reward_fn,
        environment=environment,
        config_path=config_path,
    )
    
    # Wrap for single drone (PPO models are trained with this wrapper)
    wrapped_env = SingleDroneWrapper(env)
    
    print(f"Running {episodes} episode(s)...")
    print("=" * 60)
    
    saved_logs = []
    results = []
    
    for episode in range(episodes):
        obs, info = wrapped_env.reset(seed=seed)
        done = False
        step = 0
        total_reward = 0.0
        
        while not done:
            # Get action from policy
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            done = terminated or truncated
            total_reward += reward
            step += 1
            
            if step % 100 == 0:
                distance = info.get("goal_distance", [0.0])[0] if isinstance(info.get("goal_distance"), np.ndarray) else 0.0
                print(f"  Episode {episode + 1}, Step {step}: reward={total_reward:.2f}, distance={distance:.2f}m")
        
        # Episode finished
        reason = info.get("termination_reason", "unknown")
        distance = info.get("goal_distance", [0.0])[0] if isinstance(info.get("goal_distance"), np.ndarray) else 0.0
        
        print(f"\nEpisode {episode + 1} finished:")
        print(f"  Termination: {reason}")
        print(f"  Steps: {step}")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Final distance: {distance:.2f}m")
        
        results.append({
            "episode": episode + 1,
            "termination": reason,
            "steps": step,
            "reward": total_reward,
            "distance": distance,
        })
        
        # Save episode logs
        if save_runs and env.logger:
            env.save_episode_logs()
            # Find the most recent simulation run file (episode_XXXXXX.json)
            # Note: episode_results.json is a summary file, not a simulation run file
            log_dir = Path(env.logger.log_dir)
            if log_dir.exists():
                # Filter to only match episode_NNNNNN.json files (exclude episode_results.json)
                # The pattern episode_*.json would match both, so we need to filter
                all_json_files = list(log_dir.glob("episode_*.json"))
                simulation_run_files = [
                    f for f in all_json_files 
                    if f.name != "episode_results.json" and f.name.startswith("episode_")
                ]
                
                if simulation_run_files:
                    # Sort by modification time, most recent first
                    log_files = sorted(simulation_run_files, key=lambda p: p.stat().st_mtime, reverse=True)
                    saved_logs.append(log_files[0])
                    print(f"  Saved log: {log_files[0]}")
                else:
                    print(f"  Warning: No simulation run files found in {log_dir}")
                    if all_json_files:
                        print(f"    Found JSON files: {[f.name for f in all_json_files]}")
                    print(f"    (Simulation run files should be named episode_XXXXXX.json)")
        
        print()
    
    # Print summary
    print("=" * 60)
    print("Summary:")
    for result in results:
        print(f"  Episode {result['episode']}: {result['termination']} "
              f"({result['steps']} steps, reward={result['reward']:.2f}, distance={result['distance']:.2f}m)")
    
    if saved_logs:
        print(f"\nSaved {len(saved_logs)} episode log(s) for visualization")
    
    return saved_logs


def main():
    parser = argparse.ArgumentParser(
        description="Run a trained RL policy on an environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "model_path",
        type=Path,
        help="Path to trained PPO model (.zip file)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Environment spec name (default: from config or 'large_obstacles_only')",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config file (default: config.yml in project root)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to run (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for environment (default: None)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save simulation runs (faster, but no visualization)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Automatically open visualization after running (requires --episodes=1)",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="plotly",
        choices=["plotly", "pyvista"],
        help="Visualization backend (default: plotly)",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.visualize and args.episodes > 1:
        print("Warning: --visualize only works with --episodes=1. Visualizing first episode only.")
        args.episodes = 1
    
    try:
        # Run policy
        saved_logs = run_policy(
            model_path=args.model_path,
            env_spec=args.env,
            config_path=args.config,
            episodes=args.episodes,
            seed=args.seed,
            save_runs=not args.no_save,
        )
        
        # Visualize if requested
        if args.visualize and saved_logs:
            print(f"\nOpening visualization: {saved_logs[0]}")
            vis = OfflineVisualizer(
                saved_logs[0],
                render_mode=args.render_mode,
                playback_speed=250,
            )
            vis.load()
            vis.render()
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
