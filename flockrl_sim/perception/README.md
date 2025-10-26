# Perception Module

Generates observation features for each drone using LIDAR-like ray casting and sensor simulation. This module provides the sensory data that enables drones to perceive their environment and nearby obstacles for reinforcement learning and navigation.

This is a lower priority module and can be done after the rest of the simulator is completed. This module is also highly dependent on how the Obstacles team decides to represent obstacles, so it should be completed after that is finalized

# Deliverables

Minimum:
- **Ray casting implementation**: Implement `raycast()` in `raycast.py` with ray-obstacle intersection tests to simulate distance sensors
- **Sensor observation system**: Implement `PerceptionSystem.observe()` in `sensors.py` to generate per-drone sensor readings from ray casts
- **Observation packaging**: Package ray distances and hits into `SensorReading` structures compatible with Gymnasium RL environments

Additional:
- **Batch ray casting**: Implement `raycast_batch()` using vectorized NumPy operations for 10-100x performance improvement on multi-ray LIDAR
- **Neighbor detection**: Compute relative vectors to nearby drones for multi-agent coordination features
- **Unit tests**: Cover ray intersection tests, sensor configurations, and observation generation

# Key Considerations

**Ray Casting**: Distinct from Ray Tracing, ray casting is the algorithm that enables ray casting. Ray casting will be used to determine the distance to the first obstacle each ray encounters, simulating a LiDAR sensor for our drone

**Sensor Configuration**: `SensorConfig` allows customization of max range, number of rays, and field of view. This would be critical in case we need to adjust for performance.