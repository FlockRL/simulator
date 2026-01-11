# FlockRL Simulator

This repository holds the simulator FlockRL will be using to train our drones.

## Project layout

```
flockrl_sim/
  __init__.py                # Public package exports
  state.py                   # Shared SwarmState container
  simulator.py               # Core simulator API (methods use pass)
  gym_env.py                 # Gymnasium wrapper around CoreSimulator
  rewards.py                 # Reward function base class
  config.py                  # Configuration models using Pydantic
  environment/
    __init__.py
    obstacles.py             # Environment and obstacle data structures
  collision/
    __init__.py
    system.py                # Collision detection and response stubs
  perception/
    __init__.py
    sensors.py               # Observation generation scaffolding
    raycast.py
  visualization/
    __init__.py
    renderer.py              # Offline visualization placeholders
main.py                      # Placeholder entry point
INTEGRATION_NOTES.md         # Critical team coordination points
```

## Getting started

### Installation
```bash

cd /path/to/flockrl-sim

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install the package
pip install -e .
```

## Gymnasium environment (RL)

The simulator provides a Gymnasium-compatible environment (`FlockRLGymEnv`) that can be used with standard RL libraries like Stable-Baselines3, RLlib, or custom training loops. The Gym env reads simulation and logging settings from `config.yml` in the project root by default (or a custom path you provide).

### Config (required)

`FlockRLGymEnv` loads `config.yml` and expects `simulation` and `gym` sections (see `config.yml` in this repo for defaults). If you store the config elsewhere, pass `config_path`.
- Collisions are enabled by default via the `collision` section (`enable_collisions`, `restitution`). Set `collision.enable_collisions: false` if you need a no-collision sandbox run.

### Basic Usage

```python
from pathlib import Path

import numpy as np

from flockrl_sim import FlockRLGymEnv, RewardFunction, Environment

class MyRewardFunction(RewardFunction):
    def reset(self, state) -> None:
        self._last_dist = np.linalg.norm(state.pos - state.goals, axis=1)

    def compute(self, state, action, sim_info) -> np.ndarray:
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
reward_fn = MyRewardFunction()

# Create environment (reward_fn is required - see Reward Functions section)
env = FlockRLGymEnv(
    reward_fn=reward_fn,
    environment=environment,
    config_path=Path("config.yml"),  # optional if using repo root config.yml
)

# Standard Gymnasium API
obs, info = env.reset(seed=42)
# Action shape: (num_drones, 3) - 3D acceleration vectors for each drone
action = env.action_space.sample()  
obs, rewards, terminated, truncated, info = env.step(action)
# Returns: obs shape (num_drones, obs_dim), rewards shape (num_drones,)
```

**Note on Multi-Drone Format:**
- Actions: shape `(num_drones, 3)` - 3D acceleration for each drone
- Observations: shape `(num_drones, obs_dim)` - observation for each drone
- Rewards: shape `(num_drones,)` - independent reward per drone
- Configure `num_drones` in `config.yml` (default: 1)

### Environment Setup

You can build environments manually or from JSON specs. Preset specs live under `flockrl_sim/environment/specs`.

```python
from flockrl_sim import load_environment_from_spec, load_config

config = load_config()
environment = load_environment_from_spec("simple", config)  # preset name or JSON path
```

**Note on Learning Strategy:**
When learning RL, we recommend starting by overfitting to a specific, fixed environment. This helps with:
- Debugging and understanding what your agent is learning
- Building confidence with simpler problems first
- Isolating issues (algorithm vs. environment complexity)

Once you have a working solution, you can then progress to randomized environments for better generalization.

### Reward Functions

The environment requires a custom reward function. Define your own by subclassing `RewardFunction`:

```python
from flockrl_sim import FlockRLGymEnv, RewardFunction, Environment, SwarmState
import numpy as np

class MyRewardFunction(RewardFunction):
    def __init__(self, success_reward=100.0, collision_penalty=50.0, step_cost=0.1):
        self.success_reward = success_reward
        self.collision_penalty = collision_penalty
        self.step_cost = step_cost
        self._last_distances = None
    
    def reset(self, state: SwarmState) -> None:
        """Called when environment resets - track all drones."""
        self._last_distances = np.linalg.norm(state.pos - state.goals, axis=1)
    
    def compute(self, state: SwarmState, action: np.ndarray, sim_info: dict) -> np.ndarray:
        """Compute independent rewards for each drone.
        
        Returns:
            np.ndarray: Rewards of shape (N,) where N is number of drones
        """
        # Dense reward based on progress toward goal for each drone
        current_distances = np.linalg.norm(state.pos - state.goals, axis=1)
        rewards = (self._last_distances - current_distances) - self.step_cost
        
        # Terminal rewards (applied to all drones)
        if sim_info.get("termination_reason") == "success":
            rewards += self.success_reward
        elif sim_info.get("termination_reason") == "collision":
            rewards -= self.collision_penalty
        
        self._last_distances = current_distances
        return rewards

# Create environment with your reward function
environment = Environment(
    bounds=(-100, 100, -100, 100, 0, 100),
    obstacles=[],
    start_position=(0.0, 0.0, 1.0),
    goal_position=(0.0, 0.0, 10.0),
    seed=0,
)
reward_fn = MyRewardFunction(success_reward=100.0, collision_penalty=50.0)
env = FlockRLGymEnv(
    reward_fn=reward_fn,
    environment=environment,
)
```

### Episode Logging

Episode logging is configured via `config.yml` under `gym`. Set `log_dir` (string path) to enable logging. Simulation runs are automatically saved as JSON files for visualization.

```python
env = FlockRLGymEnv(
    reward_fn=my_reward_fn,
    environment=environment,
)

# Save logs manually (I recommend doing so when checkpointing your model)
env.save_logs()
```

When logging is enabled, simulation runs are automatically saved as JSON files (`episode_XXXXXX.json`) in the log directory after each episode completes. These files can be visualized using the `OfflineVisualizer` (see Visualization section).

**Important:** Set `gym.save_runs: true` in `config.yml` **only when you need data to visualize your model**, not training/eval. Saving full simulation runs for every episode during training can consume significant disk space (up to several MB per episode for long episodes). For training, keep `save_runs: false` to only save lightweight episode statistics (`episode_results.json`).

### Observation Space

The observation includes per-drone state information with shape `(num_drones, obs_dim)`:
- **Agent state**: position (3), velocity (3)
- **Goal information**: goal vector (3), goal distance (1)
- **Sensor data**: raycast ranges and hits
- **Neighbor information**: relative positions and velocities of nearby agents (up to `max_neighbors`)
  - Each drone observes other drones within sensor range

Each drone receives its own observation independently for decentralized control.

### Action Space

Actions are 3D acceleration vectors for each drone, bounded by `max_acceleration`:
```python
# Actions have shape (num_drones, 3)
# Each action is clipped to [-max_acceleration, max_acceleration] in each dimension
action = np.array([[ax, ay, az]], dtype=np.float32)  # For single drone (num_drones=1)
action = np.array([[ax1, ay1, az1], [ax2, ay2, az2]], dtype=np.float32)  # For 2 drones
```

### Integration with RL Libraries

**Stable-Baselines3:**
```python
from stable_baselines3 import PPO
from flockrl_sim import FlockRLGymEnv, Environment

environment = Environment(
    bounds=(-100, 100, -100, 100, 0, 100),
    obstacles=[],
    start_position=(0.0, 0.0, 1.0),
    goal_position=(0.0, 0.0, 10.0),
    seed=0,
)

env = FlockRLGymEnv(
    reward_fn=my_reward_fn,
    environment=environment,
)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
```

### Complete Example

See `examples/train_simple_rl.py` for a complete training example with logging and statistics.
