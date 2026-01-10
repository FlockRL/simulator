"""
PPO Evaluation Script for FlockRL Drone Navigation

This script evaluates a trained PPO agent and optionally visualizes episodes.

Usage:
    python eval_ppo.py models/ppo_simple_20231120/best/best_model.zip
    python eval_ppo.py models/ppo_simple_20231120/final_model.zip --episodes 20
    python eval_ppo.py path/to/model.zip --env medium --render
"""

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from flockrl_sim import FlockRLGymEnv, SimulationConfig
from flockrl_sim.environment import EnvironmentSpecLoader, EnvironmentBuilder


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained PPO agent")

    parser.add_argument(
        "model_path", type=str, help="Path to trained model (.zip file)"
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="simple",
        choices=["empty", "simple", "medium", "complex"],
        help="Environment to evaluate on",
    )
    parser.add_argument(
        "--max-steps", type=int, default=500, help="Maximum steps per episode"
    )
    parser.add_argument(
        "--drone-radius", type=float, default=0.5, help="Drone collision radius"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions (no exploration)",
    )
    parser.add_argument(
        "--render", action="store_true", help="Render episodes (if supported)"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed episode information"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for evaluation"
    )

    return parser.parse_args()


def create_eval_env(env_spec: str, max_steps: int, drone_radius: float):
    """Create evaluation environment."""
    # Load environment spec
    if env_spec != "empty":
        loader = EnvironmentSpecLoader()
        spec = loader.load(env_spec)
        environment = EnvironmentBuilder.from_spec(spec).build()
    else:
        environment = None

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
        success_reward=100.0,
        collision_penalty=50.0,
        step_cost=0.1,
        distance_scale=1.0,
    )

    return env


def evaluate(args):
    """Evaluate the trained model."""
    model_path = Path(args.model_path)

    print("=" * 80)
    print("FlockRL PPO Evaluation")
    print("=" * 80)
    print(f"\nModel: {model_path}")
    print(f"Environment: {args.env}")
    print(f"Episodes: {args.episodes}")
    print(f"Deterministic: {args.deterministic}")

    # Check if model exists
    if not model_path.exists():
        print(f"\nError: Model not found at {model_path}")
        return

    # Load model
    print("\nLoading model...")
    model = PPO.load(str(model_path))

    # Create environment
    print("Creating evaluation environment...")
    env = create_eval_env(args.env, args.max_steps, args.drone_radius)

    # Check for normalization stats
    vec_normalize_path = model_path.parent / "vec_normalize.pkl"
    if vec_normalize_path.exists():
        print(f"Loading normalization stats from: {vec_normalize_path}")
        # Wrap in DummyVecEnv for VecNormalize
        vec_env = DummyVecEnv([lambda: env])
        vec_env = VecNormalize.load(str(vec_normalize_path), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        use_vec_env = True
    else:
        use_vec_env = False
        print("No normalization stats found, using raw observations")

    # Evaluation statistics
    episode_rewards = []
    episode_lengths = []
    successes = 0
    collisions = 0
    timeouts = 0

    print("\n" + "=" * 80)
    print(f"Running {args.episodes} Episodes")
    print("=" * 80 + "\n")

    # Run evaluation episodes
    for episode in range(args.episodes):
        if use_vec_env:
            obs = vec_env.reset()
        else:
            obs, _ = env.reset(seed=args.seed + episode if args.seed else None)
            obs = obs.reshape(1, -1)  # Add batch dimension

        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=args.deterministic)

            # Step environment
            if use_vec_env:
                obs, reward, done, info = vec_env.step(action)
                reward = reward[0]
                done = done[0]
                info = info[0]
            else:
                obs, reward, terminated, truncated, info = env.step(action[0])
                done = terminated or truncated
                obs = obs.reshape(1, -1)

            episode_reward += reward
            episode_length += 1

        # Record statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        termination_reason = info.get("termination_reason", "unknown")
        if termination_reason == "success":
            successes += 1
            status = "✓ SUCCESS"
        elif termination_reason == "collision":
            collisions += 1
            status = "✗ COLLISION"
        elif termination_reason == "timeout":
            timeouts += 1
            status = "⧖ TIMEOUT"
        else:
            status = f"? {termination_reason}"

        # Print episode summary
        print(f"Episode {episode + 1:2d}: {status:12s} | "
              f"Reward: {episode_reward:7.2f} | "
              f"Steps: {episode_length:3d} | "
              f"Goal dist: {info.get('goal_distance', -1):.2f}m")

        # Verbose output
        if args.verbose:
            n_collisions = len(info.get("collisions", []))
            if n_collisions > 0:
                print(f"           Collisions: {n_collisions}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    print(f"\nEpisodes: {args.episodes}")
    print(f"Success rate: {successes}/{args.episodes} ({100*successes/args.episodes:.1f}%)")
    print(f"Collision rate: {collisions}/{args.episodes} ({100*collisions/args.episodes:.1f}%)")
    print(f"Timeout rate: {timeouts}/{args.episodes} ({100*timeouts/args.episodes:.1f}%)")

    print(f"\nReward statistics:")
    print(f"  Mean: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Min:  {np.min(episode_rewards):.2f}")
    print(f"  Max:  {np.max(episode_rewards):.2f}")

    print(f"\nEpisode length statistics:")
    print(f"  Mean: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"  Min:  {np.min(episode_lengths)}")
    print(f"  Max:  {np.max(episode_lengths)}")

    # Clean up
    if use_vec_env:
        vec_env.close()
    else:
        env.close()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
