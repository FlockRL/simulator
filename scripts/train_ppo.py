#!/usr/bin/env python3
"""
Train a PPO policy on a FlockRL environment.

All settings come from config.yml. Each run creates an experiment directory
under experiments/<name>/ with a snapshot of the config and all outputs.

Usage:
    python scripts/train_ppo.py my_experiment                         # basic run
    python scripts/train_ppo.py my_experiment --config my_config.yml  # different config
"""

import argparse
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from flockrl_sim import (
    FlockRLGymEnv,
    ProgressReward,
    SingleDroneWrapper,
    load_config,
    load_environment_from_spec,
)
from flockrl_sim.gym_logging import EpisodeLogger


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class TerminationStatsCallback(BaseCallback):
    """Tracks and logs termination reasons per rollout."""

    def __init__(self, log_every: int = 50):
        super().__init__()
        self.log_every = log_every
        self.episode_counts = Counter()
        self._episodes = 0
        self._rollout_episodes = Counter()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        if not infos or dones is None:
            return True

        if isinstance(dones, np.ndarray):
            dones = dones.tolist()

        for done, info in zip(dones, infos):
            if not done or not isinstance(info, dict):
                continue
            reason = info.get("termination_reason")
            if reason is None:
                continue

            if reason == "collision":
                collisions = info.get("collisions", [])
                wall = any(getattr(c, "collision_type", None) == "wall" for c in collisions)
                obstacle = any(getattr(c, "collision_type", None) in ("clutter", "sphere") for c in collisions)
                if wall:
                    reason = "collision_wall"
                elif obstacle:
                    reason = "collision_obstacle"

            self.episode_counts[reason] += 1
            self._rollout_episodes[reason] += 1
            self._episodes += 1
            if self._episodes % self.log_every == 0:
                print(f"[{self._episodes} eps] {dict(self.episode_counts)}")
        return True

    def on_rollout_end(self) -> None:
        total = sum(self._rollout_episodes.values())
        self.logger.record("termination/episodes_total", float(self._episodes))
        self.logger.record("termination/rollout_episodes", float(total))
        if total > 0:
            for reason, count in self._rollout_episodes.items():
                tag = reason.replace("/", "_").replace(" ", "_")
                self.logger.record(f"termination/rollout_rate/{tag}", count / total)
            print(f"  Rollout: {dict(self._rollout_episodes)} ({total} eps)")
        self._rollout_episodes.clear()


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_env(rank: int, config: dict, config_path: Path, exp_dir: Path):
    """Create a single environment instance for parallel training."""
    def _init():
        reward_fn = ProgressReward.from_config(config)
        environment = load_environment_from_spec(config["environment"]["spec"], config)

        flockrl_env = FlockRLGymEnv(
            reward_fn=reward_fn,
            environment=environment,
            config_path=config_path,
        )
        env = SingleDroneWrapper(flockrl_env)

        if rank == 0:
            flockrl_env.logger = EpisodeLogger(log_dir=exp_dir)

        env = Monitor(env, str(exp_dir / f"monitor_{rank}"), info_keywords=("termination_reason",))
        return env
    return _init


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train PPO on FlockRL")
    parser.add_argument("name", type=str, help="Experiment name (saved under experiments/<name>/)")
    parser.add_argument("--config", type=Path, default=Path("config.yml"), help="Path to config.yml")
    args = parser.parse_args()

    # Load config
    config_path = args.config.resolve()
    config = load_config(config_path)
    train_cfg = config["training"]
    ppo_cfg = train_cfg["ppo"]

    total_timesteps = int(train_cfg["total_timesteps"])
    num_envs = int(train_cfg["num_envs"])

    # Create experiment directory
    env_spec = config["environment"]["spec"]
    name = args.name
    exp_dir = Path("experiments") / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)

    # Snapshot config into experiment directory
    snapshot_path = exp_dir / "config.yml"
    shutil.copy2(config_path, snapshot_path)

    print(f"Experiment: {exp_dir}")
    print(f"Env spec:   {env_spec}")
    print(f"Timesteps:  {total_timesteps:,}")
    print(f"Num envs:   {num_envs}")
    print("=" * 60)

    # Create vectorized environment (use snapshot so config_path matches exp_dir)
    env = SubprocVecEnv([make_env(i, config, snapshot_path, exp_dir) for i in range(num_envs)])

    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=float(ppo_cfg["learning_rate"]),
        n_steps=int(ppo_cfg["n_steps"]),
        batch_size=int(ppo_cfg["batch_size"]),
        n_epochs=int(ppo_cfg["n_epochs"]),
        gamma=float(ppo_cfg["gamma"]),
        gae_lambda=float(ppo_cfg["gae_lambda"]),
        clip_range=float(ppo_cfg["clip_range"]),
        ent_coef=float(ppo_cfg["ent_coef"]),
        vf_coef=float(ppo_cfg["vf_coef"]),
        max_grad_norm=float(ppo_cfg["max_grad_norm"]),
        policy_kwargs={
            "log_std_init": float(ppo_cfg["log_std_init"]),
            "ortho_init": bool(ppo_cfg["ortho_init"]),
        },
        verbose=1,
        tensorboard_log=str(exp_dir / "tensorboard"),
    )

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=int(train_cfg["checkpoint_freq"]),
        save_path=str(exp_dir / "checkpoints"),
        name_prefix="ppo",
    )
    stats_cb = TerminationStatsCallback()

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_cb, stats_cb],
        progress_bar=True,
    )

    # Save final model
    model_path = exp_dir / "model.zip"
    model.save(str(model_path))
    print(f"\nModel saved: {model_path}")
    print(f"Experiment:  {exp_dir}")
    print("\nTo run this policy:")
    print(f"  python scripts/run_ppo.py {exp_dir}")


if __name__ == "__main__":
    sys.exit(main() or 0)
