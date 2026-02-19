# RL Training Guide

This guide covers the PPO training workflow in this repository, including setup, training, monitoring with TensorBoard, evaluation, and common troubleshooting.

## Overview

The RL flow uses:

- `scripts/train_ppo.py` to train a policy with Stable-Baselines3 PPO
- `scripts/monitor_training.py` to launch TensorBoard for an experiment
- `scripts/run_ppo.py` to evaluate a trained model and optionally visualize runs
- `config.yml` as the single source of truth for environment, reward, and PPO settings

Each training run creates an experiment directory at `experiments/<name>/`.

## 1) Environment Setup

Install project dependencies and RL extras:

```bash
uv sync --extra rl
```

## 2) Configure Training

Training and reward settings live in `config.yml`:

- `environment.spec`: environment spec file name in `flockrl_sim/environment/specs`
  - For your training, you should focus on: `2_obstacles`, `3_obstacles`, `rand_2_obstacles`, `rand_3_obstacles`. These are more representive of what our IRL portion will look like
  - You can visualize these environments with `examples/interactive_viewer.py` to see obstacle layouts, spawn/goal positions, and bounds

## 3) Start Training

Run training with an experiment name:

```bash
uv run python scripts/train_ppo.py exp-1
```

All training settings come from `config.yml`. The script creates an experiment directory and saves a snapshot of the config used for that run.

Outputs are written to:

- `experiments/exp-1/config.yml` (snapshot used for the run)
- `experiments/exp-1/checkpoints/` (periodic checkpoints)
- `experiments/exp-1/tensorboard/` (TensorBoard events)
- `experiments/exp-1/model.zip` (final model)

## 4) Monitor Training (TensorBoard)

Monitor a specific run:

```bash
uv run python scripts/monitor_training.py exp-1
```

List experiments:

```bash
uv run python scripts/monitor_training.py --list
```

Then open `http://localhost:6006`.

## 5) Evaluate a Trained Policy

Run the final model:

```bash
uv run python scripts/run_ppo.py exp-1
```

Run multiple episodes:

```bash
uv run python scripts/run_ppo.py exp-1 -n 5
```

Use a specific checkpoint:

```bash
uv run python scripts/run_ppo.py exp-1 --model checkpoints/ppo_100000_steps.zip
```

Override environment spec at eval time:

```bash
uv run python scripts/run_ppo.py exp-1 --env rand_3_obstacles
```

Visualize one run:

```bash
uv run python scripts/run_ppo.py exp-1 --viz
```

Evaluation logs are saved in `experiments/<name>/eval_runs/`.

## 6) Reward Notes

The default reward is progress-based (`ProgressReward`):

- Positive reward when distance-to-goal decreases
- Per-step penalty to encourage efficiency
- Success bonus when goal is reached
- Collision penalties (wall/obstacle/fallback)

Core step reward term:

`reward = progress_scale * (last_dist - curr_dist) - step_penalty + alive_bonus`

## 7) Suggested Workflow

1. Tune `config.yml`: tune or completely redesign rewards, and even specs if you want to experiment (use `examples/interactive_viewer.py` to help you tune the specs if needed)
2. Train: `uv run python scripts/train_ppo.py <exp_name>`
3. Monitor: `uv run python scripts/monitor_training.py <exp_name>`
4. Evaluate: `uv run python scripts/run_ppo.py <exp_name> -n 5`
5. Compare runs in `experiments/` and iterate
