"""
Simple RL training example showing the Gymnasium API with logging.

This script demonstrates:
1. Creating a FlockRL Gym environment with logging enabled
2. Using custom reward functions
3. Using the standard Gymnasium API (with PPO from Stable-Baselines3)
4. Tracking episode outcomes and statistics
5. Exporting results for analysis
"""

import numpy as np
from typing import Any, Dict
from pathlib import Path
from flockrl_sim import FlockRLGymEnv, RewardFunction, SwarmState, load_environment_from_spec, load_config
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

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
        if sim_info["termination_reason"] == "success":
            rewards += self.success_reward
        elif sim_info["termination_reason"] == "collision":
            rewards -= self.collision_penalty

        self._last_distances = current_distances
        return rewards


class TrainingCallback(BaseCallback):
    """
    Custom callback to track episode statistics during PPO training.
    """
    def __init__(self, print_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.episode_count = 0
        self.recent_results = []
        
    def _on_step(self) -> bool:
        # Check if episode ended
        for idx, done in enumerate(self.locals.get("dones", [])):
            if done:
                self.episode_count += 1
                
                # Get episode info if available
                if "infos" in self.locals and idx < len(self.locals["infos"]):
                    info = self.locals["infos"][idx]
                    if "episode_result" in info:
                        self.recent_results.append(info["episode_result"])
                        
                        # Print progress every print_freq episodes
                        if self.episode_count % self.print_freq == 0:
                            recent = self.recent_results[-self.print_freq:]
                            if recent:
                                successes = sum(1 for r in recent if r.termination_reason == "success")
                                collisions = sum(1 for r in recent if r.termination_reason == "collision")
                                success_rate = successes / len(recent)
                                collision_rate = collisions / len(recent)
                                avg_reward = sum(r.total_reward for r in recent) / len(recent)
                                print(
                                    f"Episode {self.episode_count:4d}: "
                                    f"Success={success_rate:6.1%}, "
                                    f"Collision={collision_rate:6.1%}, "
                                    f"Avg Reward={avg_reward:7.2f}"
                                )
        return True


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

    # Check if checkpoint exists
    checkpoint_dir = Path("./logs/checkpoints/")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_checkpoint = None
    
    # Find the latest checkpoint
    checkpoints = list(checkpoint_dir.glob("rl_model_*_steps.zip"))
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[2]))
        print(f"Found checkpoint: {latest_checkpoint}")
        print("Loading model from checkpoint...\n")
        model = PPO.load(latest_checkpoint, env=env)
        print("✓ Model loaded successfully!")
    else:
        # Create PPO model
        print("No checkpoint found. Creating new PPO model...")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Encourage exploration
            verbose=1,
            # tensorboard_log="./logs/tensorboard/"  # Install tensorboard to enable: pip install tensorboard
        )
    
    # Create callbacks
    tracking_callback = TrainingCallback(print_freq=10, verbose=1)  # Print every 10 episodes
    
    # Checkpoint callback - saves model every 5000 timesteps
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="./logs/checkpoints/",
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    
    # Combine callbacks
    callbacks = [tracking_callback, checkpoint_callback]
    
    # Train the model
    print("\nStarting PPO training...")
    print("Training for 50,000 timesteps (approximately 50 episodes)...")
    print("Checkpoints will be saved every 5000 timesteps\n")
    
    total_timesteps = 50_000  # Quick training for testing
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=False)
    
    # Save the final trained model
    final_model_path = "logs/ppo_flockrl_final"
    model.save(final_model_path)
    print(f"\n✓ Final trained model saved to: {final_model_path}")
    print(f"✓ Checkpoints saved in: {checkpoint_dir}")
    print(f"\nTo resume training, just run this script again!")
    print(f"To load the model: model = PPO.load('{final_model_path}')")
    
    # Test the trained policy for a few episodes
    print("\nTesting trained policy...")
    test_episodes = 10
    for episode in range(test_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward  # reward is now a scalar
            done = terminated or truncated
        
        print(f"Test Episode {episode + 1}: Reward={episode_reward:.2f}, Outcome={info['termination_reason']}")

    # Save final logs
    print("\n" + "=" * 60)
    print("Saving logs to disk...")
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
