#!/usr/bin/env python3
"""
Run a trained PPO policy from an experiment directory.

Uses the config.yml snapshot saved during training so the environment
matches exactly what the model was trained on. Always saves flight logs
for visualization regardless of config settings.

The experiment argument can be a name (searched in experiments/) or a full path.

Usage:
    python scripts/run_ppo.py my_run                                              # run 1 episode
    python scripts/run_ppo.py my_run -n 5                                        # run 5 episodes
    python scripts/run_ppo.py my_run --viz                                       # run and visualize
    python scripts/run_ppo.py my_run --env rand_3_obstacles                      # different env
    python scripts/run_ppo.py my_run --model checkpoints/ppo_100000_steps.zip   # specific checkpoint
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import PPO

from flockrl_sim import (
    FlockRLGymEnv,
    ProgressReward,
    SingleDroneWrapper,
    load_config,
    load_environment_from_spec,
)
from flockrl_sim.visualization import OfflineVisualizer


def run(exp_dir: Path, model_rel: str, env_spec: str | None, episodes: int, seed: int | None, visualize: bool):
    exp_dir = exp_dir.resolve()
    config_path = exp_dir / "config.yml"
    if not config_path.exists():
        sys.exit(f"No config.yml in {exp_dir}. Is this an experiment directory?")

    model_path = exp_dir / model_rel
    if not model_path.exists():
        sys.exit(f"Model not found: {model_path}")

    config = load_config(config_path)

    # Override env spec if requested
    if env_spec:
        config["environment"]["spec"] = env_spec
    spec = config["environment"]["spec"]

    # Force save_runs on and set log_dir so we always get flight output
    run_dir = exp_dir / "eval_runs"
    run_dir.mkdir(exist_ok=True)
    config["gym"]["save_runs"] = True
    config["gym"]["log_dir"] = str(run_dir)

    # Write patched config so FlockRLGymEnv reads the correct values
    eval_config_path = run_dir / "config.yml"
    with open(eval_config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Experiment: {exp_dir.name}")
    print(f"Model:      {model_rel}")
    print(f"Env spec:   {spec}")
    print(f"Episodes:   {episodes}")
    print("=" * 60)

    model = PPO.load(str(model_path))

    reward_fn = ProgressReward.from_config(config)
    environment = load_environment_from_spec(spec, config)

    flockrl_env = FlockRLGymEnv(
        reward_fn=reward_fn,
        environment=environment,
        config_path=eval_config_path,
    )
    env = SingleDroneWrapper(flockrl_env)

    saved_logs: list[Path] = []

    for ep in range(episodes):
        ep_seed = seed + ep if seed is not None else None
        obs, info = env.reset(seed=ep_seed)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        reason = info.get("termination_reason", "unknown")
        dist = info.get("goal_distance", np.array([0.0]))
        dist = float(dist[0]) if isinstance(dist, np.ndarray) else float(dist)

        print(f"  Episode {ep+1}: {reason} | {steps} steps | reward={total_reward:.1f} | dist={dist:.2f}m")

        # Collect saved episode file
        if flockrl_env.logger:
            flockrl_env.save_episode_logs()
            episode_files = sorted(run_dir.glob("episode_[0-9]*.json"), key=lambda p: p.stat().st_mtime)
            if episode_files:
                saved_logs.append(episode_files[-1])

    print("=" * 60)

    if visualize and saved_logs:
        log_file = saved_logs[0]
        print(f"Visualizing: {log_file.name}")
        vis = OfflineVisualizer(log_file, render_mode="plotly", playback_speed=250)
        vis.load()
        vis.render()

    return saved_logs


def resolve_experiment(name: str) -> Path:
    """Resolve experiment name to a directory, searching experiments/ if needed."""
    p = Path(name)
    if p.exists():
        return p
    candidate = Path("experiments") / name
    if candidate.exists():
        return candidate
    # Fuzzy: look for any subdir containing the name
    experiments_dir = Path("experiments")
    if experiments_dir.is_dir():
        matches = sorted(experiments_dir.iterdir())
        exact = [d for d in matches if d.name == name]
        if exact:
            return exact[0]
        partial = [d for d in matches if name in d.name]
        if len(partial) == 1:
            return partial[0]
        if len(partial) > 1:
            sys.exit(f"Ambiguous experiment name '{name}'. Matches:\n" + "\n".join(f"  {d.name}" for d in partial))
    sys.exit(f"Experiment '{name}' not found (checked ./{name} and experiments/{name})")


def main():
    parser = argparse.ArgumentParser(description="Run a trained PPO policy")
    parser.add_argument("experiment", type=str, help="Experiment name (in experiments/) or full path")
    parser.add_argument("--model", type=str, default="model.zip", help="Model file relative to experiment dir (default: model.zip)")
    parser.add_argument("--env", type=str, default=None, help="Override environment spec")
    parser.add_argument("-n", "--episodes", type=int, default=1, help="Number of episodes (default: 1)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--viz", action="store_true", help="Visualize first episode after running")
    args = parser.parse_args()

    exp_dir = resolve_experiment(args.experiment)
    run(exp_dir, args.model, args.env, args.episodes, args.seed, args.viz)


if __name__ == "__main__":
    main()
