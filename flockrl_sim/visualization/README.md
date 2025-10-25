# Visualization Module

Provides offline visualization capabilities for simulation playback and analysis. This module will load simulation logs from disk and render 3D visualizations of drone trajectories, obstacles, and collision events for post-simulation analysis.

# Deliverables

Minimum:
- **3D Visualization Window**: Render drones and obstacles in a 3D view
- **Offline Data Processing**: Load and process saved simulation data for static visualization

Additional (in order of priority):
- **Control Panel**: UI for simulation replay (step control, playback speed, zoom/rotate)
- **Path Visualization**: Display drone trajectories as paths in 3D space
- **Unit Tests**: Verify preprocessing layer, data streaming, and visualization outputs
- **Online Data Processing**: Process current simulation data for dynamic real-time visualization
- **Drone POV View (stretch goal)**: Optional per-drone camera perspective if there's subteam interest in extending the visualizer

# Ideas

Focus only on the position from each drone, you will likely not need to know the velocity and acceleration of each drone

Attempt visualizing the movement of a single drone first without any obstacles before visualizing with obstacles. This way you won't be blocked by the obstacles team as they will be starting roughly when you start too

If time allows, experimenting with a POV-style view could be a fun extension, but it is not required for the initial milestone.

# Future Plans

Use an online simulator approach 
