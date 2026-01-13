# Offline Visualization

The visualization module loads JSON logs produced by `CoreSimulator.save_run()` and renders them with a selectable backend. The entry point is `OfflineVisualizer` in `renderer.py`.

## Data flow

- `OfflineVisualizer.load()` reads the log file, stores `frames` and `metadata`, and extracts obstacles from `metadata.environment.obstacles` (or `metadata.obstacles` if present).
- `OfflineVisualizer.render()` selects either the Plotly or PyVista backend and delegates rendering.

## Plotly backend

`plotly_renderer.py` builds an interactive Dash app with:

- A 3D scene containing drone markers, obstacle meshes, and translucent goal spheres.
- A frame slider and play/pause interval for animation.
- Goal radius derived from `metadata["config"]["simulation"]["goal_threshold"]`.

This backend imports `dash` and `plotly` at runtime and raises a `RuntimeError` if they are missing.

## PyVista backend

`pyvista_renderer.py` renders a 3D scene using PyVista with:

- Point cloud updates for drone positions.
- Line segments drawn between consecutive frames to show trajectories.
- A simple floor plane derived from environment bounds when available.

This backend imports `pyvista` at runtime and raises a `RuntimeError` if it is missing.

## Log structure

`save_run()` writes:

```json
{
  "metadata": { "environment": { "obstacles": [...] }, "config": { ... } },
  "frames": [
    { "state": { "pos": [...], "vel": [...], "acc": [...], "ids": [...], "goals": [...] },
      "info": { "collisions": [...], "observations": [...] } }
  ]
}
```
