# FlockRL Simulator Skeleton

This repository holds the simulator FlockRL will be using to train our drones.

## Project layout

```
flockrl_sim/
  __init__.py                # Public package exports
  state.py                   # Shared SwarmState container
  simulator.py               # Core simulator API (methods use pass)
  config.py                  # Configuration models using Pydantic
  environment/
    __init__.py
    obstacles.py             # Environment and obstacle data structures
  collision/
    __init__.py
    system.py                # Collision detection and response stubs
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

### Team assignments

- **Core Simulation & Kinematics**: Implement `simulator.py` and `state.py` methods
- **Obstacles**: Expand `environment/obstacles.py` with Wall, Gate, Sphere, Box classes
- **Collision Handling**: Implement `collision/system.py` detection and response logic
- **Visualization**: Implement `visualization/renderer.py` for offline playback

### Important notes

1. **No runnable logic yet** - All methods are stubs with `pass`
2. **Read INTEGRATION_NOTES.md** - Critical coordination points between teams
3. **Interface changes** - Coordinate with affected teams before modifying shared interfaces
4. **Type hints** - All interfaces have type hints for IDE support
5. **Configs use Pydantic** - Validation and defaults built-in

### Data flow

```
action → CoreSimulator.step() → kinematic update → 
  CollisionSystem() → collision detection/response → 
    updated state → CoreSimulator.log_frame() → 
      CoreSimulator.save_run() (writes SimulationRun to disk) → 
        OfflineVisualizer (reads and renders)
```

The simulator maintains a `SimulationRun` object internally and writes it to disk using `CoreSimulator.save_run()` so that the visualization team can replay runs offline via `OfflineVisualizer` (in `visualization/renderer.py`).

Each file contains docstrings pointing to the relevant feature area so collaborators know where to add their work.
