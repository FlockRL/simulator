# Environment Module

Defines the simulation environment including obstacle types, boundaries, and spatial configuration. This module generates and provides data structures for walls with gates and environment bounds that serve as the foundation for collision detection and visualization systems.

# Definitions

- Boundaries: Outer boundary of the simulator
- Walls: A series of walls, each should have a gate (hole) that allows a drone to pass through
- Gates: Holes in walls where drones can pass through
- Clutter: Random objects that can future complicate drone movement

# Deliverables

Minimum:
- **Config-based initialization**: Build environments from structured configs using `EnvironmentConfig`, including preset or randomly generated spawn zones, wall spacing, gate placement/size, and support for multiple obstacle types (walls, gates, spheres, boxes, clutter) with full geometry metadata.
- **Pre-flight validation**: Run validations (no overlaps, gates fully embedded, etc.) during environment construction to ensure valid configurations.
- **Testing & validation**: Provide unit tests for individual obstacle definitions, integration tests that ensure reachable paths and valid spawn points, and surface scene metrics (counts, density) so downstream teams can gauge environment difficulty.

# Ideas

We can have a each gate act as an "obstacle" that is inside of each wall obstacle. While a drone that is touching a wall is considered a collision, we can ignore this collision if and only if the drone is fully inside the gate "obstacle" 
