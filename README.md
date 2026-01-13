# FlockRL Simulator

This repository contains the `flockrl_sim` Python package used for FlockRL training and offline analysis. The implementation centers on a fixed-timestep physics step, collision response against bounds and obstacles, optional perception sampling, and JSON run logging for visualization.

## Runtime pipeline

- `FlockRLGymEnv` loads `config.yml`, constructs `CoreSimulator` and `CollisionSystem`, and exposes the Gymnasium API.
- `CoreSimulator.step()` validates actions, integrates velocity/position with constant acceleration, runs collision detection, applies per-drone corrections, updates episode stats, and checks termination (goal, bounds, timeout, collision).
- If perception is enabled, `PerceptionSystem.observe()` produces raycast ranges/hits and neighbor vectors, which are packed into the observation tensor in `FlockRLGymEnv`.

## Package layout (implementation)

```
flockrl_sim/
  __init__.py                Public API exports
  state.py                   SwarmState container and cloning
  simulator.py               CoreSimulator, SimulationRun/Frame, JSON logging
  gym_env.py                 Gymnasium wrapper and observation assembly
  gym_logging.py             EpisodeLogger/EpisodeResult
  rewards.py                 RewardFunction base class
  geometry.py                OBB math and intersection helpers
  environment/
    obstacles_types.py       Obstacle dataclasses + ray_intersect implementations
    obstacles.py             Environment, EnvironmentBuilder
    loader.py                EnvironmentSpecLoader
    spec_models/             Pydantic models + random value resolvers
      environment.py         EnvironmentSpec model
      obstacles.py           WallSpec, ClutterSpec, GateSpec models
      random_values.py       Random value resolution (uniform, discrete)
    validation.py            Geometry/overlap/gate checks
    specs/                   JSON environment presets
  collision/
    system.py                CollisionSystem and CollisionInfo
  perception/
    sensors.py               PerceptionSystem, SensorConfig/Reading
    raycast.py               Raycast helpers
  visualization/
    renderer.py              OfflineVisualizer entry point
    plotly_renderer.py       Dash/Plotly renderer
    pyvista_renderer.py      PyVista renderer
```

## Documentation

- `docs/gym-env-usage.md` Gymnasium environment usage, configuration, and logging
- `docs/core-simulator.md` Core simulator engine and logging format
- `docs/collision-system.md` Collision detection and response details
- `docs/perception-system.md` Raycasting and neighbor observation logic
- `docs/environment-model.md` Environment data model and spec build flow
- `docs/environment-specs.md` JSON spec schema and validation rules
- `docs/offline-visualization.md` Offline visualizer backends and log structure

## Environment specs

Preset environments live in `flockrl_sim/environment/specs`. Specs are validated with Pydantic models, resolved into concrete obstacles by `EnvironmentBuilder`, and checked for bounds, overlaps, and gate embedding before being returned as an `Environment` instance.

## Logging and visualization

- `EpisodeLogger` writes lightweight episode summaries to `episode_results.json` when enabled.
- If `gym.save_runs` is true and `gym.log_dir` is set, `CoreSimulator.save_run()` writes full frame data as JSON with metadata (environment, obstacles, and config) for offline rendering for each episode.
- `OfflineVisualizer` loads these JSON logs and dispatches to the Plotly or PyVista backends.
