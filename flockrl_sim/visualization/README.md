# Visualization Module

The visualization module provides offline visualization capabilities for simulation playback and analysis. It loads simulation logs from disk and renders interactive 3D visualizations of drone trajectories, obstacles, goals, and collision events for post-simulation analysis.

## Features

- **3D Interactive Visualization**: Render drones, obstacles, and goals in an interactive 3D environment
- **Trajectory Visualization**: Display complete drone flight paths as animated trails
- **Playback Controls**: Play, pause, reset, and scrub through simulation frames
- **Multiple Renderers**: Choose between Plotly (web-based) or PyVista (desktop) backends
- **Real-time Information**: Display frame number, simulation time, and drone count
- **Customizable Playback Speed**: Adjust animation speed to analyze fast or slow motion

## Installation

The visualization module requires additional dependencies depending on which renderer you want to use:

### Plotly Renderer
```bash
pip install dash plotly
```

### PyVista Renderer
```bash
pip install pyvista
```

## Quick Start

### Basic Usage

```python
from pathlib import Path
from flockrl_sim.visualization import OfflineVisualizer

# Load and render a JSON log file
log_path = Path("demo_simulation_output.json")
vis = OfflineVisualizer(log_path, playback_speed=250)
vis.load()
vis.render()
```

### Using the Example Script

```bash
python examples/visualize_json_logs.py path/to/log.json
```

If no path is provided, it defaults to `demo_simulation_output.json` in the project root.

## Renderers

The module supports two rendering backends:

### Plotly Renderer (Default)

**Usage:**
```python
vis = OfflineVisualizer(log_path, render_mode="plotly", playback_speed=250)
vis.load()
vis.render(host="127.0.0.1", port=8050, debug=False)
```

### PyVista Renderer

**Usage:**
```python
vis = OfflineVisualizer(log_path, render_mode="pyvista", playback_speed=250)
vis.load()
vis.render(window_size=(1920, 1080))
```

## API Reference

### `OfflineVisualizer`

Main class for loading and visualizing simulation logs.

#### Constructor

```python
OfflineVisualizer(
    log_path: Path,
    render_mode: str = "plotly",
    playback_speed: float = 250.0
)
```

**Parameters:**
- `log_path` (Path): Path to the JSON log file saved by `CoreSimulator.save_run()`
- `render_mode` (str): Rendering backend to use. Options: `"plotly"` (default) or `"pyvista"`
- `playback_speed` (float): Milliseconds between frames during playback (default: 250ms)

#### Methods

##### `load() -> None`

Loads the simulation log from the JSON file and extracts obstacles from metadata.

**Raises:**
- `FileNotFoundError`: If the log file doesn't exist
- `ValueError`: If the log file contains no frames

##### `render(mode: Optional[str] = None, **kwargs) -> None`

Renders the loaded simulation using the selected backend.

**Parameters:**
- `mode` (Optional[str]): Override the render mode. If None, uses the mode set in constructor.
- `**kwargs`: Additional arguments passed to the renderer:
  - For Plotly: `host` (str), `port` (int), `debug` (bool)
  - For PyVista: `window_size` (Tuple[int, int])

**Raises:**
- `RuntimeError`: If no data has been loaded (call `load()` first)
- `ValueError`: If an unknown render mode is specified

## Usage Examples

### Example 1: Basic Visualization

```python
from pathlib import Path
from flockrl_sim.visualization import OfflineVisualizer

log_path = Path("logs/simple_training/episode_000000.json")
vis = OfflineVisualizer(log_path)
vis.load()
vis.render()
```

### Example 2: Fast Playback with PyVista

```python
from pathlib import Path
from flockrl_sim.visualization import OfflineVisualizer

log_path = Path("demo_simulation_output.json")
vis = OfflineVisualizer(log_path, render_mode="pyvista", playback_speed=100)
vis.load()
vis.render(window_size=(2560, 1440))
```

### Example 3: Custom Plotly Server

```python
from pathlib import Path
from flockrl_sim.visualization import OfflineVisualizer

log_path = Path("demo_simulation_output.json")
vis = OfflineVisualizer(log_path, render_mode="plotly", playback_speed=500)
vis.load()
vis.render(host="0.0.0.0", port=8080, debug=True)
```

## Log File Format

The visualizer expects JSON log files created by `CoreSimulator.save_run()` with the following structure:

```json
{
  "metadata": {
    "obstacles": [...],
    "environment": {...}
  },
  "frames": [
    {
      "state": {
        "pos": [[x, y, z], ...],
        "ids": [0, 1, 2, ...],
        "goals": [[x, y, z], ...],
        "t": 0.0
      }
    },
    ...
  ]
}
```

## Controls

### Plotly Renderer Controls

- **Play/Pause Button**: Start or pause the animation
- **Reset Button**: Jump back to frame 0
- **Frame Slider**: Manually scrub to any frame in the simulation
- **Speed Slider**: Adjust playback speed (50-1000ms between frames)
- **Mouse Controls**:
  - Left-click + drag: Rotate the 3D view
  - Right-click + drag: Pan the view
  - Scroll: Zoom in/out

### PyVista Renderer Controls

- **Checkbox Button**: Toggle animation on/off (green = playing, red = stopped)
- **Mouse Controls**: Standard 3D viewport controls (rotate, zoom, pan)
- **Keyboard**: Standard PyVista keyboard shortcuts apply

## Troubleshooting

### "ModuleNotFoundError: No module named 'dash'"

Install the required dependencies:
```bash
pip install dash plotly
```

### "ModuleNotFoundError: No module named 'pyvista'"

Install PyVista:
```bash
pip install pyvista
```

### "No frames found in log file"

Ensure your log file was created using `CoreSimulator.save_run()` and contains a `frames` array with at least one frame.

### Plotly server won't start

- Check if the port is already in use (default: 8050)
- Try specifying a different port: `vis.render(port=8051)`
- Ensure you have network permissions if binding to `0.0.0.0`

### PyVista window doesn't appear

- Ensure you have a display available (PyVista requires a GUI)
- On headless systems, use the Plotly renderer instead
- Check that PyVista is properly installed: `python -c "import pyvista; print(pv.__version__)"`

## Future Enhancements

Potential future improvements include:
- Online/real-time visualization during simulation
- Per-drone camera perspective (POV view)
- Collision event highlighting
- Export to video formats
- Multi-episode comparison views 
