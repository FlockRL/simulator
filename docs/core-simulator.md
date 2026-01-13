# Core Simulator Engine

This module implements the simulation loop, state bookkeeping, and run logging. The main entry point is `CoreSimulator`, which integrates motion, applies collision responses, computes termination, and records per-step metadata.

## Key types

- `SwarmState` (`flockrl_sim/state.py`): arrays for positions, velocities, accelerations, drone IDs, and goals. The class provides cloning and a `from_initial_positions` constructor.
- `SimulationFrame` and `SimulationRun` (`flockrl_sim/simulator.py`): container types used for offline playback and JSON serialization.
- `CoreSimulator` (`flockrl_sim/simulator.py`): orchestrates the physics step and episode lifecycle.

## Step flow

`CoreSimulator.step()` follows a fixed sequence:

1. Validate action shape, sanitize NaNs/Infs, and clip by `max_acceleration` if configured.
2. Integrate with constant acceleration:
   - `vel = vel + acc * delta_t`
   - `pos = pos + vel * delta_t + 0.5 * acc * delta_t^2`
   - `t = t + delta_t`
3. Call the collision system (callable) to detect contacts.
4. Apply per-drone position corrections and rebound velocities derived from `CollisionInfo`. For drones with multiple simultaneous collisions, position corrections are accumulated additively, and velocity rebounds are applied sequentially in the normal direction to properly handle corner collisions.
5. Update episode stats and check termination conditions.
6. Optionally compute perception observations and log a frame.

## Termination conditions

Episodes terminate when any of these conditions is met (checked in priority order):

1. **Collision**: If `terminate_on_collision` is enabled and any collision occurred this step
2. **Timeout**: Step count reaches `max_steps`
3. **Success**: All drones are within `goal_threshold` distance of their respective goals
4. **Out of bounds**: Any drone position is outside the environment bounds (this should be prevented by bounds collision detection, but is checked as a safety measure)

## Episode management

- `start_run()` seeds a new `SimulationRun` and records the initial frame.
- `reset()` can return to the original initial state or add Gaussian noise using `reset_position_noise` and `reset_velocity_noise` from `reset_config`.
- Termination reasons are stored in `info` and tracked in `_episode_stats`.

## Logging format

`save_run()` writes JSON with two top-level keys:

- `metadata`: arbitrary metadata plus optional environment/obstacle data.
- `frames`: a list of `{state, info}` snapshots, with collision and perception data serialized to plain lists.
