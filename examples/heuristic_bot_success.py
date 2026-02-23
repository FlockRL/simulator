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


def obstacle_aware_policy(obs: np.ndarray, config: dict) -> np.ndarray:
    """
    Improved policy: move toward goal while avoiding obstacles.
    
    Observation structure (for single drone):
    - vel (3): velocity  
    - goal_vector (3): vector from position to goal
    - goal_distance (1): distance to goal
    - ranges (num_rays): distance readings from raycasts
    - hits (num_rays): binary hit indicators (1 if obstacle detected)
    - neighbors (max_neighbors * 6): relative position + velocity per neighbor
    """
    if obs.ndim == 2:
        obs = obs[0]  # Handle (1, obs_dim) shape
    
    # Extract components
    vel = obs[0:3]
    goal_vector = obs[3:6]
    goal_distance = obs[6]
    
    # Extract sensor data
    num_rays = config["perception"]["num_rays"]
    ranges = obs[7:7+num_rays]
    hits = obs[7+num_rays:7+2*num_rays]
    
    # Base goal direction (normalized)
    goal_direction = goal_vector / (goal_distance + 1e-6)
    
    # Obstacle avoidance using potential field approach
    avoidance_force = np.zeros(3, dtype=np.float32)
    
    # Parameters
    danger_threshold = 3.5  # Start avoiding obstacles within this distance (meters)
    avoidance_strength = 2.5  # How strongly to avoid obstacles
    
    # Get ray angles (assuming uniform distribution around the drone)
    angles = np.linspace(0, 2*np.pi, num_rays, endpoint=False)
    
    for i in range(num_rays):
        if hits[i] > 0.5:  # Obstacle detected
            distance = ranges[i]
            
            if distance < danger_threshold:
                # Ray direction in 3D (horizontal plane, assuming rays are 2D)
                ray_dir = np.array([np.cos(angles[i]), np.sin(angles[i]), 0.0])
                
                # Repulsive force inversely proportional to distance
                # Closer obstacles create stronger repulsion
                repulsion_magnitude = avoidance_strength * (1.0 - distance / danger_threshold)
                
                # Push away from obstacle (opposite to ray direction)
                avoidance_force -= ray_dir * repulsion_magnitude
    
    # Combine goal attraction and obstacle repulsion
    desired_direction = goal_direction + avoidance_force
    
    # Normalize to unit vector (the environment will scale by max_acceleration)
    direction_magnitude = np.linalg.norm(desired_direction)
    if direction_magnitude > 1e-6:
        desired_direction = desired_direction / direction_magnitude
    else:
        # If stuck, just try moving toward goal
        desired_direction = goal_direction
    
    return desired_direction.reshape(1, 3).astype(np.float32)

def main():
    """Run simple heuristic bot demo."""
    # Load config from demo-specific YAML file
    config_path = Path(__file__).parent / "heuristic_bot_success.yaml"
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
        action = obstacle_aware_policy(obs, config)
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
