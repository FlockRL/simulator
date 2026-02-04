"""
Test and visualize a trained PPO policy.

This script loads a trained model and runs test episodes with visualization enabled.
"""

import numpy as np
from pathlib import Path
import yaml
from stable_baselines3 import PPO
from flockrl_sim import FlockRLGymEnv, RewardFunction, SwarmState, load_environment_from_spec, load_config


class SimpleRewardFunction(RewardFunction):
    """Simple reward function (same as training)."""
    
    def __init__(
        self,
        success_reward: float = 100.0,
        collision_penalty: float = 50.0,
        step_cost: float = 0.1,
    ):
        self.success_reward = success_reward
        self.collision_penalty = collision_penalty
        self.step_cost = step_cost
        self._last_distances = None

    def reset(self, state: SwarmState) -> None:
        """Initialize distance tracking for all drones."""
        self._last_distances = np.linalg.norm(state.pos - state.goals, axis=1)

    def compute(self, state: SwarmState, action: np.ndarray, sim_info: dict) -> np.ndarray:
        """Compute independent rewards for each drone based on progress toward goal."""
        N = state.pos.shape[0]
        current_distances = np.linalg.norm(state.pos - state.goals, axis=1)
        
        # Base reward: progress toward goal minus step cost
        rewards = (self._last_distances - current_distances) - self.step_cost

        # Terminal rewards (applied to all drones when episode ends)
        if sim_info["termination_reason"] == "success":
            rewards += self.success_reward
        elif sim_info["termination_reason"] == "collision":
            rewards -= self.collision_penalty

        self._last_distances = current_distances
        return rewards


def test_policy(model_path: str, num_episodes: int = 5, save_visualization: bool = True):
    """
    Test a trained policy and optionally save episodes for visualization.
    
    Args:
        model_path: Path to the trained model (without .zip extension)
        num_episodes: Number of test episodes to run
        save_visualization: If True, saves episodes as JSON for offline visualization
    """
    # Load configuration
    config = load_config()
    
    # Enable visualization logging if requested
    config_path = None
    if save_visualization:
        config["gym"]["log_dir"] = "logs/test_episodes"
        config["gym"]["save_runs"] = True
        
        # Write modified config to temp file
        import tempfile
        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False)
        yaml.dump(config, temp_file)
        temp_file.close()
        config_path = Path(temp_file.name)
    
    # Create reward function
    reward_fn = SimpleRewardFunction(
        success_reward=100.0, collision_penalty=50.0, step_cost=0.1
    )

    # Load environment from spec
    environment = load_environment_from_spec("simple", config)
    
    # Create environment with modified config
    env = FlockRLGymEnv(
        reward_fn=reward_fn,
        environment=environment,
        config_path=config_path,
    )

    print("=" * 60)
    print("Testing Trained PPO Policy")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Environment: simple")
    print(f"Test episodes: {num_episodes}")
    if save_visualization:
        print(f"Saving episodes to: {config['gym']['log_dir']}")
    print("=" * 60)
    print()

    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    print("✓ Model loaded successfully!\n")

    # Run test episodes
    results = []
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        
        print(f"Episode {episode + 1}/{num_episodes}:")
        
        while not done:
            # Use trained policy (deterministic mode for testing)
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
        
        # Print episode results
        result = {
            "episode": episode + 1,
            "steps": steps,
            "reward": episode_reward,
            "outcome": info["termination_reason"],
            "final_distance": info["goal_distance"][0]
        }
        results.append(result)
        
        print(f"  Steps: {steps}")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Outcome: {info['termination_reason']}")
        print(f"  Final distance to goal: {info['goal_distance'][0]:.2f}m")
        print()
    
    # Save final episode logs
    if save_visualization:
        env.save_episode_logs()
        print("=" * 60)
        print("✓ Episode data saved!")
        print(f"  Location: {config['gym']['log_dir']}")
        print(f"  Files: episode_000000.json to episode_{num_episodes-1:06d}.json")
        print()
        print("To visualize, run:")
        print(f"  python examples/visualize_json_logs.py")
        print("=" * 60)
        print()
    
    # Print summary statistics
    successes = sum(1 for r in results if r["outcome"] == "success")
    collisions = sum(1 for r in results if r["outcome"] == "collision")
    timeouts = sum(1 for r in results if r["outcome"] == "timeout")
    avg_reward = sum(r["reward"] for r in results) / len(results)
    avg_steps = sum(r["steps"] for r in results) / len(results)
    
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Total episodes:    {num_episodes}")
    print(f"Success rate:      {successes/num_episodes:.1%}")
    print(f"Collision rate:    {collisions/num_episodes:.1%}")
    print(f"Timeout rate:      {timeouts/num_episodes:.1%}")
    print(f"Avg reward:        {avg_reward:.2f}")
    print(f"Avg steps:         {avg_steps:.1f}")
    print("=" * 60)
    
    # Clean up temp config file if created
    if config_path and config_path.exists():
        import os
        os.unlink(config_path)
    
    return results


if __name__ == "__main__":
    # Test the final trained model
    model_path = "logs/ppo_flockrl_final"
    
    # You can also test a specific checkpoint:
    # model_path = "logs/checkpoints/rl_model_10000_steps"
    
    # Run tests with visualization enabled
    results = test_policy(
        model_path=model_path,
        num_episodes=5,
        save_visualization=True
    )
