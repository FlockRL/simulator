# Core Simulator Module

The main simulation engine that orchestrates the entire flocking simulation. This module manages the simulation loop, integrates collision detection, and provides logging capabilities

# Deliverables

Minimum:
- **Simulator Class**: Python class to encapsulate simulation logic and state
- **State Initialization**: Method to set up initial SwarmState (position, velocity, acceleration, IDs)
- **Kinematic step() Function**: Advances simulation by one time step using physics equations
- **Fixed Timestep Management**: Maintain a consistent `delta_t` for stable integration

Additional:
- **Demo Script**: Demonstrate drone movement with hardcoded controls
- **Unit Tests**: Verify kinematic calculations, edge cases, and multi-drone states

# Ideas

Model drones as point masses. We can just have the collision team inform us if drone collides with anything as the collision team can just worry about the shape.

# Future Ideas

Consider rotational mechanics in the future. However, don't focus on this for now, as it would require rethinking the entire control approach.

**Why rotation is out of scope:**

Currently, drones are controlled via a 3-element acceleration vector (x, y, z directions). This means no rotational forces are applied to each drone. 

Adding rotation would require:
- Redefining actions (e.g., controlling four individual rotors)
- Modeling the drone's body and rotational dynamics
- Significantly more complexity

This would shift the project from RL-focused to simulator-focused.
