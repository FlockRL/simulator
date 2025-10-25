# Collision Module

Handles collision detection and response for drone swarms in the simulation environment. This module provides collision detection algorithms for walls, gates, clutter objects, and inter-drone collisions

# Deliverables

Minimum:
- **Static wall collision detection**: Distance to planes, penetration depth, contact normal
- **Gate collision detection**: Block movement through walls unless the drone is passing through a gate
- **Clutter collision detection**: Detect collisions with spheres, cubes, and other geometric shapes
- **Inter-drone collision handling**: Detect and resolve collisions between drones so they cannot overlap

Additional:
- **Collision rebounding**: Simulate elastic collisions using collision point, direction, and momentum
- **Unit tests**: Cover wall collisions, gate collisions, clutter collisions, and rebounding logic

# Key Considerations

**Drone Representation**: Drones are represented as point masses with `[x, y, z]` coordinates. You will need to implement shape detection logic around this point to determine collisions with obstacles. For now, treat drones as a sphere centered around the point mass.

# Ideas

You will need to collaborate closely with the obstacles team to determine how they want to represent obstacles

We can have a each gate act as an "obstacle" that is inside of each wall obstacle. While a drone that is touching a wall is considered a collision, we can ignore this collision if and only if the drone is fully inside the gate "obstacle" 
