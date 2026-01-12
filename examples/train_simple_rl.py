"""
Simple RL training example showing the Gymnasium API with logging.

This script demonstrates:
1. Creating a FlockRL Gym environment with logging enabled
2. Using custom reward functions
3. Using the standard Gymnasium API
4. Tracking episode outcomes and statistics
5. Exporting results for analysis
"""

import numpy as np
from typing import Any, Dict
from flockrl_sim import FlockRLGymEnv, RewardFunction, SwarmState, load_environment_from_spec, load_config
import pandas as pd

class SimpleRewardFunction(RewardFunction):
    """
    Simple dense reward function for training.
    
    Computes independent rewards for each drone based on their progress toward their goal.
    Each drone gets its own reward signal for independent learning.
    """

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

    def compute(
        self, state: SwarmState, action: np.ndarray, sim_info: Dict[str, Any]
    ) -> np.ndarray:
        """Compute independent rewards for each drone based on progress toward goal."""
        N = state.pos.shape[0]
        current_distances = np.linalg.norm(state.pos - state.goals, axis=1)
        
        # Base reward: progress toward goal minus step cost
        rewards = (self._last_distances - current_distances) - self.step_cost

        # Terminal rewards (applied to all drones when episode ends)
        if sim_info.get("termination_reason") == "success":
            rewards += self.success_reward
        elif sim_info.get("termination_reason") == "collision":
            rewards -= self.collision_penalty

        self._last_distances = current_distances
        return rewards


def random_policy(obs: np.ndarray) -> np.ndarray:
    """
    Dummy policy for demonstration purposes.

    In a real training script, this would be replaced with your RL policy
    (e.g., from Stable-Baselines3, RLlib, or your own implementation).

    Args:
        obs: Observation from environment, shape (N, obs_dim) where N is number of drones

    Returns:
        Random action in the action space, shape (N, 3)
    """
    num_drones = obs.shape[0]
    return np.random.uniform(-5, 5, size=(num_drones, 3))


def main():
    # Load configuration
    config = load_config()
    
    # Create reward function
    reward_fn = SimpleRewardFunction(
        success_reward=100.0, collision_penalty=50.0, step_cost=0.1
    )

    # Load environment from spec
    environment = load_environment_from_spec("simple", config)
    # Create environment with logging enabled
    env = FlockRLGymEnv(
        reward_fn=reward_fn,
        environment=environment,
    )

    print("=" * 60)
    print("FlockRL Simple Training Example")
    print("=" * 60)
    print(f"Environment: simple")
    print(f"Bounds: {environment.bounds}")
    print(f"Obstacles: {len(environment.obstacles)}")
    print(f"Start position: {environment.start_position}")
    print(f"Goal position: {environment.goal_position}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    print(f"Logging to: logs/simple_training")
    print("=" * 60)
    print()

    # Train for N episodes
    num_episodes = 1000

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False

        while not done:
            action = random_policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # Print progress every 100 episodes
        if (episode + 1) % 100 == 0 and env.logger:
            results = list(env.logger._results)[-100:]
            if results:
                successes = sum(1 for r in results if r.termination_reason == "success")
                collisions = sum(1 for r in results if r.termination_reason == "collision")
                success_rate = successes / len(results)
                collision_rate = collisions / len(results)
                print(
                    f"Episode {episode + 1:4d}: "
                    f"Success={success_rate:6.1%}, "
                    f"Collision={collision_rate:6.1%}"
                )

    # Save final logs
    print("\nSaving logs to disk...")
    env.save_episode_logs()

    # Print summary statistics
    if env.logger:
        results = list(env.logger._results)
        if results:
            total = len(results)
            successes = sum(1 for r in results if r.termination_reason == "success")
            collisions = sum(1 for r in results if r.termination_reason == "collision")
            timeouts = sum(1 for r in results if r.termination_reason == "timeout")
            avg_steps = sum(r.steps for r in results) / total
            avg_reward = sum(r.total_reward for r in results) / total
            avg_goal_dist = sum(r.final_goal_distance for r in results) / total
            min_goal_dist = min(r.min_goal_distance for r in results)
            max_reward = max(r.total_reward for r in results)

            print("\n" + "=" * 60)
            print("Training Summary")
            print("=" * 60)
            print(f"Total episodes:       {total}")
            print(f"Success rate:         {successes/total:.1%}")
            print(f"Collision rate:       {collisions/total:.1%}")
            print(f"Timeout rate:         {timeouts/total:.1%}")
            print(f"Avg episode length:   {avg_steps:.1f} steps")
            print(f"Avg total reward:     {avg_reward:.2f}")
            print(f"Avg goal distance:    {avg_goal_dist:.2f} m")
            print(f"Min goal dist achieved: {min_goal_dist:.2f} m")
            print(f"Max reward achieved:  {max_reward:.2f}")
            print("=" * 60)

            # Export to CSV for analysis
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
            csv_path = "logs/episode_results.csv"
            df.to_csv(csv_path, index=False)
            print(f"\nEpisode statistics exported to: {csv_path}")
            print(f"You can analyze this data with pandas, Excel, or other tools.")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
