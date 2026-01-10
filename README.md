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

# Now you can use flockrl from anywhere (while venv is active)
flockrl generate -n 10 -o output.json
```

### Command-Line Interface

```bash
# Show help
flockrl --help

# Generate sample simulation data
flockrl generate -n 10 -f 200 -o my_simulation.json

# All options
flockrl generate -n 5 -f 100 -d 10.0 -s 42 -o output.json
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

from flockrl_sim import FlockRLGymEnv, RewardFunction
from flockrl_sim.environment import Environment

class MyRewardFunction(RewardFunction):
    def reset(self, state) -> None:
        self._last_dist = float(np.linalg.norm(state.pos[0] - state.goals[0]))

    def compute(self, state, action, sim_info) -> float:
        current = float(np.linalg.norm(state.pos[0] - state.goals[0]))
        reward = self._last_dist - current
        self._last_dist = current
        return reward

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
action = env.action_space.sample()  # 3D acceleration vector
obs, reward, terminated, truncated, info = env.step(action)
```

### Environment Setup

You can build environments manually or from JSON specs. Preset specs live under `flockrl_sim/environment/specs`.

```python
from flockrl_sim.environment import EnvironmentBuilder, EnvironmentSpecLoader

spec = EnvironmentSpecLoader().load("simple")  # preset name or JSON path
environment = EnvironmentBuilder.from_spec(spec).config
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
from flockrl_sim import FlockRLGymEnv, RewardFunction
from flockrl_sim.environment import Environment
from flockrl_sim.state import SwarmState
import numpy as np

class MyRewardFunction(RewardFunction):
    def __init__(self, success_reward=100.0, collision_penalty=50.0, step_cost=0.1):
        self.success_reward = success_reward
        self.collision_penalty = collision_penalty
        self.step_cost = step_cost
        self._last_distance = 0.0
    
    def reset(self, state: SwarmState) -> None:
        """Called when environment resets."""
        self._last_distance = float(np.linalg.norm(state.pos[0] - state.goals[0]))
    
    def compute(self, state: SwarmState, action: np.ndarray, sim_info: dict) -> float:
        """Compute reward for current step."""
        # Dense reward based on progress toward goal
        current_dist = float(np.linalg.norm(state.pos[0] - state.goals[0]))
        reward = (self._last_distance - current_dist) - self.step_cost
        
        # Terminal rewards
        if sim_info.get("termination_reason") == "success":
            reward += self.success_reward
        elif sim_info.get("termination_reason") == "collision":
            reward -= self.collision_penalty
        
        self._last_distance = current_dist
        return reward

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

Episode logging is configured via `config.yml` under `gym`. Set `log_dir` (string path) to enable logging, and `save_trajectories` to store full trajectories.

```python
env = FlockRLGymEnv(
    reward_fn=my_reward_fn,
    environment=environment,
)

# Save logs manually (I recommend doing so when checkpointing your model)
env.save_logs()
```

**Trajectory Logging:**
For detailed analysis, set `gym.save_trajectories: true` in `config.yml`:
```python
env = FlockRLGymEnv(reward_fn=my_reward_fn, environment=environment)
```

### Observation Space

The observation includes:
- **Agent state**: position (3), velocity (3)
- **Goal information**: goal vector (3), goal distance (1)
- **Sensor data**: raycast ranges and hits
- **Neighbor information**: relative positions and velocities of nearby agents (up to `max_neighbors`)
  - Current Gym wrapper runs a single drone; neighbor features are zero-padded.

### Action Space

Actions are 3D acceleration vectors bounded by `max_acceleration`:
```python
# Action is clipped to [-max_acceleration, max_acceleration] in each dimension
action = np.array([ax, ay, az], dtype=np.float32)
```

### Integration with RL Libraries

**Stable-Baselines3:**
```python
from stable_baselines3 import PPO
from flockrl_sim import FlockRLGymEnv
from flockrl_sim.environment import Environment

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
