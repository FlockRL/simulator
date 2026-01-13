# Collision System

This module implements collision detection and response for drones represented as spheres. The `CollisionSystem` is callable and returns a list of `CollisionInfo` entries describing each contact.

## CollisionInfo

Each contact includes:
- `drone_id`
- `collision_type` ("bounds", "wall", "clutter", "sphere", or "drone")
- `normal_vector`, `contact_point`, `penetration_depth`
- `rebound_velocity` and `new_position`

`CoreSimulator` applies these corrections per drone to resolve overlaps and update velocities.

## Detection paths

- **Bounds**: checks each drone sphere against the configured environment bounds; collisions are generated per violated face.
- **Drones**: checks pairwise sphere-sphere overlap between drones and resolves both drones symmetrically.
- **Walls and gates**: walls are treated as rectangular prisms. Collisions are skipped if the drone sphere fits within the gate's width and height dimensions, and the drone center is within the gate's thickness (gates are typically thin pass-through regions, so only the center needs to be within the thickness dimension).
- **Clutter**: rectangular prisms (subtype `rectangular_prism`) are tested against the drone sphere.
- **Spheres**: any obstacle with a `radius` attribute is treated as a sphere for collision checks.

## Geometry and response

Rectangular prism collisions use oriented bounding boxes (OBB). If the prism has zero rotation, the implementation falls back to a dedicated AABB path; otherwise it uses OBB math from `flockrl_sim/geometry.py`. Rebound velocity is computed by reflecting the normal component and scaling it by the coefficient of restitution.

## Multiple simultaneous collisions

When a drone experiences multiple collisions in a single step (e.g., hitting a corner or colliding with multiple objects), the `CoreSimulator` handles them by:

1. **Position corrections**: Accumulated additively from all collisions
2. **Velocity rebounds**: Applied sequentially in each collision's normal direction

This approach ensures that corner collisions and complex multi-contact scenarios are resolved correctly without requiring iterative collision resolution.
