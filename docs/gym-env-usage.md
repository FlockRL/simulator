# Gymnasium Environment Usage

`FlockRLGymEnv` wraps `CoreSimulator` and exposes the Gymnasium API for RL training. It loads `config.yml`, builds a collision system, optionally enables perception, and emits observations shaped for multi-drone policies.

## Basic setup

```python
from pathlib import Path
import numpy as np

from flockrl_sim import FlockRLGymEnv, RewardFunction, Environment

class MyReward(RewardFunction):
    def reset(self, state):
        self._last_dist = np.linalg.norm(state.pos - state.goals, axis=1)

    def compute(self, state, action, sim_info):
        current = np.linalg.norm(state.pos - state.goals, axis=1)
        rewards = self._last_dist - current
        self._last_dist = current
        return rewards

environment = Environment(
    bounds=(-100, 100, -100, 100, 0, 100),
    obstacles=[],
    start_position=(0.0, 0.0, 1.0),
    goal_position=(0.0, 0.0, 10.0),
    seed=0,
)

env = FlockRLGymEnv(
    reward_fn=MyReward(),
    environment=environment,
    config_path=Path("config.yml"),
)

obs, info = env.reset(seed=42)
action = env.action_space.sample()
obs, rewards, terminated, truncated, info = env.step(action)
```

## Environment presets

To build environments from JSON specs, use `load_environment_from_spec` and the config settings for spawn clearance:

```python
from flockrl_sim import load_environment_from_spec, load_config

config = load_config()
env = load_environment_from_spec("simple", config)
```

## Config fields used

`FlockRLGymEnv` reads these sections from `config.yml`:

- `gym`: `num_drones`, `spawn_offset_range`, `max_neighbors`, `log_dir`, `save_runs`
- `simulation`: `delta_t`, `max_steps`, `goal_threshold`, `max_acceleration`, `terminate_on_collision`, `reset_position_noise`, `reset_velocity_noise`
- `collision`: `drone_radius`, `restitution`
- `perception`: `max_range`, `num_rays`, `max_neighbour_range`
- `visualization`: `fps`

Reset noise is applied on every `reset()` using `reset_position_noise` and `reset_velocity_noise` (set them to `0.0` to disable).

## Observation layout

Per drone, the observation vector concatenates:

- position (3)
- velocity (3)
- goal vector (3)
- goal distance (1)
- ray ranges (`num_rays`)
- ray hits (`num_rays`)
- neighbor vectors (`max_neighbors * 6`)

The final tensor shape is `(num_drones, obs_dim)`.

## Action layout

Actions are per-drone accelerations with shape `(num_drones, 3)`. Actions are clipped to `[-max_acceleration, max_acceleration]`.

## Episode logging

If `gym.log_dir` is set, `EpisodeLogger` will store summary statistics. Call `env.save_episode_logs()` to flush `episode_results.json`. If `gym.save_runs` is `true` (and `gym.log_dir` is set), full frame logs are saved as `episode_XXXXXX.json` files, suitable for `OfflineVisualizer`.
