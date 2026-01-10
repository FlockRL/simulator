"""
Quick test to verify PPO training setup works correctly.
Runs a very short training session (10k steps) to check for errors.

Usage:
    python test_training.py
"""

import shutil
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from flockrl_sim import FlockRLGymEnv, SimulationConfig


def test_training():
    """Run a quick training test."""
    print("=" * 70)
    print("PPO Training Test")
    print("=" * 70)

    # Create test directories
    test_dir = Path("test_training_run")
    if test_dir.exists():
        print(f"\nCleaning up old test directory: {test_dir}")
        shutil.rmtree(test_dir)

    log_dir = test_dir / "logs"
    model_dir = test_dir / "models"
    log_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)

    print(f"\nTest directory: {test_dir}")

    # Create environment
    print("\n1. Creating environment...")
    env = make_vec_env(
        lambda: FlockRLGymEnv(
            sim_config=SimulationConfig(max_steps=100),
            enable_collisions=True,
        ),
        n_envs=2,  # Just 2 parallel envs for testing
        vec_env_cls=DummyVecEnv,
    )
    print("   ✓ Environment created")

    # Create PPO model
    print("\n2. Creating PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=256,  # Small for quick test
        batch_size=32,
        verbose=1,
        tensorboard_log=str(log_dir),
    )
    print("   ✓ Model created")

    # Train for a short time
    print("\n3. Running short training (10k steps)...")
    print("   (This should take ~30 seconds)")
    model.learn(total_timesteps=10000, progress_bar=True)
    print("   ✓ Training completed")

    # Save model
    print("\n4. Saving model...")
    model_path = model_dir / "test_model"
    model.save(str(model_path))
    print(f"   ✓ Model saved to: {model_path}.zip")

    # Load and test model
    print("\n5. Loading and testing saved model...")
    loaded_model = PPO.load(str(model_path))
    print("   ✓ Model loaded successfully")

    # Quick evaluation
    print("\n6. Running quick evaluation (3 episodes)...")
    eval_env = FlockRLGymEnv()
    successes = 0
    total_reward = 0

    for ep in range(3):
        obs, _ = eval_env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _ = loaded_model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_reward += episode_reward
        if info.get("termination_reason") == "success":
            successes += 1

        result = "✓" if info.get("termination_reason") == "success" else "✗"
        print(f"   Episode {ep+1}: {result} {info.get('termination_reason'):10s} "
              f"Reward: {episode_reward:6.1f}")

    print(f"\n   Success rate: {successes}/3")
    print(f"   Average reward: {total_reward/3:.1f}")
    print("   ✓ Evaluation completed")

    # Clean up
    print("\n7. Cleaning up test directory...")
    shutil.rmtree(test_dir)
    print("   ✓ Cleanup complete")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70)
    print("\nYour training setup is working correctly!")
    print("\nNext steps:")
    print("  1. Run a full training session:")
    print("     python train_ppo.py --total-steps 1e6")
    print("\n  2. Monitor progress:")
    print("     tensorboard --logdir logs/")
    print("\n  3. Evaluate trained model:")
    print("     python eval_ppo.py models/ppo_*/best/best_model.zip")


if __name__ == "__main__":
    try:
        test_training()
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
