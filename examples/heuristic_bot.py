"""
Simple heuristic bot demo for FlockRL gym environment.

This demonstrates basic usage of the gym environment with simple controls:
just accelerate toward the goal.

Pay close attention to the config file and how logging is configured.
"""

import numpy as np
from pathlib import Path
from flockrl_sim import FlockRLGymEnv, RewardFunction, SwarmState, load_environment_from_spec, load_config


class SimpleRewardFunction(RewardFunction):
    """Simple reward: progress toward goal."""

    def __init__(self):
        self._last_distance = None

    def reset(self, state: SwarmState) -> None:
        self._last_distance = np.linalg.norm(state.pos[0] - state.goals[0])

    def compute(self, state: SwarmState, action: np.ndarray, sim_info: dict) -> np.ndarray:
        current_distance = np.linalg.norm(state.pos[0] - state.goals[0])
        reward = self._last_distance - current_distance  # Progress reward
        self._last_distance = current_distance
        return np.array([reward])


def simple_policy(obs: np.ndarray, config: dict) -> np.ndarray:
    """
    Super simple policy: just move toward the goal.
    
    Observation structure (for single drone):
    - vel (3): velocity  
    - goal_vector (3): vector from position to goal
    - goal_distance (1): distance to goal
    - ... (sensor data, neighbors, etc.)
    """
    if obs.ndim == 2:
        obs = obs[0]  # Handle (1, obs_dim) shape
    
    # Extract goal vector and distance (indices 3-6 after vel)
    goal_vector = obs[3:6]
    goal_distance = obs[6]
    
    # Normalize to unit direction vector (environment handles scaling)
    goal_direction = goal_vector / (goal_distance + 1e-6)
    return goal_direction.reshape(1, 3).astype(np.float32)


def main():
    """Run simple heuristic bot demo."""
    # Load config from demo-specific YAML file
    config_path = Path(__file__).parent / "heuristic_bot.yaml"
    config = load_config(config_path)
    
    environment = load_environment_from_spec("obstacles_only", config)
    
    # Create reward function
    reward_fn = SimpleRewardFunction()
    
    # Create gym environment
    env = FlockRLGymEnv(
        reward_fn=reward_fn,
        environment=environment,
        config_path=config_path,
    )
    
    print("FlockRL Simple Heuristic Bot Demo")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    if env.logger:
        print(f"Logging to: {env.logger.log_dir}")
    print()
    
    obs, info = env.reset()
    done = False
    step = 0
    
    while not done:
        action = simple_policy(obs, config)
        obs, reward, terminated, truncated, info = env.step(action) # Note that the reward is ignored in this example
        done = terminated or truncated
        step += 1
        
        if step % 50 == 0:
            distance = info["goal_distance"][0]
            print(f"Step {step}: distance to goal = {distance:.2f}m")
    
    # Print result
    reason = info["termination_reason"]
    distance = info["goal_distance"][0]
    print(f"Episode finished: {reason}, final distance = {distance:.2f}m")
    
    # Save logs
    if env.logger:
        env.save_episode_logs()
        print(f"Logs saved to: {env.logger.log_dir}")

if __name__ == "__main__":
    main()
