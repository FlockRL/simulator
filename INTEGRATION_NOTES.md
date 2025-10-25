# Team Integration Notes

This document outlines critical coordination points between feature teams to ensure smooth integration.

## Obstacles <-> Collision Teams

**Critical coordination required:**

- **Obstacles team** defines obstacle geometry (Wall, Gate, Sphere, Box, etc.)
- **Collision team** needs to query obstacle geometry for detection
- Both teams must agree on:
  - How obstacle surfaces/boundaries are represented
  - What methods/properties obstacles expose for distance queries
  - How gate openings are checked

## Collision <-> Core Simulation

**When collisions occurs:**

- Collision team creates a collision event and inform the core simulation team that an event has occured
- Collision team will also update the new drone velocities (rebound) and position (to prevent re-penetration) and inform the core simulation team of the new velocity via each collision event
- For moving obstacles (future): coordinate on obstacle velocity updates

## Core Simulation ↔ All Teams

**Ownership:**

- Core Simulation owns `SwarmState` (the single source of truth for drone states)
- Core Simulation owns `Environment` (the single source of truth for obstacle data)
- All teams **read** from `SwarmState` and `Environment`
- Only **Core Simulation** will **write** to `SwarmState`
- Only **Core Simulation** creates and manages the `Environment` instance

**Environment Management:**
- **Core Simulation** creates and owns the `Environment` instance using `EnvironmentBuilder`
- **Collision team** reads obstacle data directly from the `Environment` instance
- **Obstacles team** provides the `Environment` and `EnvironmentBuilder` classes

**Integration flow:**

For each frame:
```
Step 1: Core Simulation:
  1. Applies kinematic update based on actions
  2. Passes proposed state to Collision system
  
Step 2: Collision:
  3. Detects collisions using Environment data
  4. Creates a collision event with new velocity/position if collisions occur
  5. Returns collision info dict
  
Step 3: Core Simulation
  Option 1: Proceed with simulation on collisions:
    6. Uses collision events info, if any, to adjust drone positions/velocities
    7. Logs frame using CoreSimulator.log_frame() (appends to SimulationRun) 
  
  Option 2: End simulation on collisions:
    6. Ends simulation if there's any collision events
```

## Core Simulation ↔ Visualization

**Data flow:**

- Core Simulation maintains a `SimulationRun` object and writes it using `CoreSimulator.save_run()`
- Visualization uses `OfflineVisualizer` (in `visualization/renderer.py`) to read logs

**What Visualization needs:**

- `SimulationRun` files saved by `CoreSimulator.save_run()`
- Obstacle data from `Environment` (geometry, positions)
- Collision events from info dict (collision markers, normals)
- Drone states from `SwarmState` history (positions, velocities for trajectory)

**Integration points:**

- Logging is built into `CoreSimulator` (via `log_frame()` and `save_run()`)
- Visualization only reads and renders, doesn't create logs
- Offline visualization doesn't block the simulation loop
- Online visualization (future) would hook into `CoreSimulator.render_hook`

## Testing Integration Points

**Recommended integration tests:**

1. **Obstacles + Collision**: Place a drone on collision course with a wall, verify collision is detected and velocity is updated
2. **Core Simulation + Collision**: Run a multi-step simulation, verify state updates are consistent
3. **Full pipeline**: Generate environment -> run simulation -> save log -> load and visualize

## Communication

If you need to modify a shared interface (e.g., adding fields to `SwarmState`, changing `Obstacle` structure), please:

1. Document the change in a design doc or issue
2. Notify affected teams
3. Update this integration notes document
4. Coordinate any breaking changes
