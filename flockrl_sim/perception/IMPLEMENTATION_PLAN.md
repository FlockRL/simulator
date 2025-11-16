# Perception Team Implementation Plan

This document extends the [README.md](README.md) with detailed requirements and implementation guidance for the perception module.

## Summary

Obstacles aren't fully defined yet, so we will be using a mock_obstacles class for perception development.

## Interface Contract

All obstacles (mock or real) implement the following interface:

```python
def ray_intersect(
    self, 
    origin: np.ndarray,      # shape=(3,)
    direction: np.ndarray,   # shape=(3,), normalized
    max_distance: float
) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    """
    Returns: (distance, hit_point, normal) or None if no hit
    """
```

## Requirements

### Minimum Deliverables

#### 1. `raycast()` Function (in `raycast.py`)

**Purpose**: Cast a single ray and find the closest obstacle intersection to simulate distance sensors.

**Inputs**:
- `origin`: Starting point of the ray (3D position, shape=(3,))
- `direction`: Normalized direction vector of the ray (shape=(3,))
- `obstacles`: List of obstacles to check against
- `max_distance`: Maximum distance to search for intersections

**Outputs**:
- `RayHit` tuple `(distance, hit_point, normal)` if an intersection is found within max_distance
- `None` if no intersection is found

**Requirements**:
- Must iterate through all obstacles in the provided list
- For each obstacle, call `obstacle.ray_intersect(origin, direction, max_distance)` to check for intersections
- Must track and return the closest intersection if multiple intersections exist
- Must respect max_distance limit (ignore intersections beyond this distance)
- Must handle cases where no intersection occurs
- Must work with obstacles that implement the `ray_intersect()` interface

#### 2. `PerceptionSystem.observe()` Method (in `sensors.py`)

**Purpose**: Generate sensor readings for drones based on ray casting. This will be updated with each time step using the `state`

**Inputs**:
- `state`: SwarmState containing drone positions and IDs
- Uses `self.config`: SensorConfig with `max_range`, `num_rays`, and `max_neighbour_range`
- Uses `self.environment`: Environment containing obstacles

**Outputs**:
- `List[SensorReading]`: One SensorReading per drone, maintaining the same ordering as found in state (i.e., `readings[i]` corresponds to `state.pos[i]`)

**SensorReading Structure**:
- `ranges`: Array of ray-cast distances, shape=(num_rays,)
- `hits`: Boolean array indicating whether each ray hit an obstacle, shape=(num_rays,)
- `neighbor_vectors`: Optional array of relative vectors to nearby drones (for additional deliverable)
- `metadata`: Optional dictionary for extra data (surface normals, obstacle IDs, etc.)

**Requirements**:
- Must generate ray directions based on sensor configuration (`self.config.num_rays`)
  - Ray directions must be **spherically uniform** (evenly distributed on a unit sphere)
  - Use spherical coordinate generation to create `num_rays` uniformly distributed directions
- Must process observations for all drones in the swarm state
- Must package results into `SensorReading` format matching the dataclass structure

#### 3. Observation Packaging

**Purpose**: Package ray distances and hits into `SensorReading` structures compatible with Gymnasium RL environments.

**Requirements**:
- Must create `SensorReading` objects with correct array shapes
- `ranges` array must have shape `(num_rays,)` with float values
- `hits` array must have shape `(num_rays,)` with boolean values
- Must return list of SensorReading objects maintaining the same ordering as found in state (i.e., `readings[i]` corresponds to `state.pos[i]`)

### Additional Deliverables

#### 4. Neighbor Detection (Optional)

**Purpose**: Detect nearby drones in the swarm and compute relative vectors to enable multi-agent coordination and flocking behaviors.

**What it does**:
- For each drone, identify other drones within a detection range
- Compute relative position vectors from the observing drone to each detected neighbor
- Consider computing relative velocity vectors for velocity-based coordination
- Populate the `neighbor_vectors` field in `SensorReading` with this information

**What are relative position vectors?**:
Drones are treated as points. Populate `neighbor_vectors` by computing relative vectors from the observing drone's center coordinates to each neighbor drone's center coordinates using their positions from `SwarmState.pos`.

**Inputs**:
- `state`: SwarmState containing positions, velocities, and IDs of all drones
- Uses `self.config.max_neighbour_range`: Maximum distance to consider a drone as a "neighbor" (separate from `max_range` for ray casting)

**Outputs**:
- `neighbor_vectors`: Array of relative vectors to nearby drones
  - Shape: `(num_neighbors, 6)` for positions and velocities
  - Each vector represents the relative position and velocity from the observing drone to a neighbor
  - Empty array if no neighbors detected within range

#### 5. Batch Ray Casting (`raycast_batch()`) (Optional)

**Purpose**: Cast multiple rays efficiently using vectorized operations for performance optimization.

**Requirements**:
- Implement `raycast_batch()` function in `raycast.py`
- Use vectorized operations to handle multiple rays efficiently
- Provide performance improvement for scenarios with many rays per drone (128+)
- Function signature is already defined in `raycast.py`

## Implementation Phases

### Phase 1: Core Ray Casting and Sensor Integration
- Implement `raycast()` function in `raycast.py`
- Implement `PerceptionSystem.observe()` method in `sensors.py`

### Phase 2: Integration
- Replace mock obstacles with real `Environment.obstacles` when available
- Verify compatibility with real obstacle system