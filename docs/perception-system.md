# Perception System

The perception module generates per-drone observations using raycasting and neighbor queries. The entry point is `PerceptionSystem.observe()`, which returns a `SensorReading` for each drone.

## SensorConfig and SensorReading

- `SensorConfig` defines `max_range`, `num_rays`, and `max_neighbour_range`.
- `SensorReading` holds `ranges`, `hits`, `neighbor_vectors`, plus optional metadata.
- `RayHit` is a tuple type `(distance, hit_point, obstacle)` representing a ray intersection result.

## Ray generation and filtering

- `generate_rays()` samples `num_rays` unit vectors uniformly over the sphere using a seeded RNG.
- For each drone and each ray, the system calls `Obstacle.ray_intersect` and keeps the nearest hit.
- Gate volumes are treated as transparent: wall hits inside a gate are ignored, and direct gate hits are discarded.

## Neighbor vectors

Neighbor data is computed from pairwise relative positions and velocities. Drones beyond `max_neighbour_range` are excluded, and the remaining neighbors are ordered by distance. Each neighbor contributes a 6D vector `[dx, dy, dz, dvx, dvy, dvz]`.

## Helper functions

`raycast()` provides a standalone closest-hit helper, while `raycast_batch()` is a stub-style helper that expects obstacles to implement a batch ray intersection method. The current implementation in `PerceptionSystem` performs per-ray scalar intersection calls.
