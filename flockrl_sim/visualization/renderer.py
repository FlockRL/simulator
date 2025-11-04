"""
Provides offline visualization capabilities for simulation playback and analysis.
This module will load simulation logs from disk and render 3D visualizations of drone trajectories for post-simulation analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd
import pyvista as pv
import json

from ..simulator import SimulationRun


class OfflineVisualizer:
    """
    Loads simulation logs from disk and produces offline visualizations.
    """

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.run: Optional[SimulationRun] = None
        self.data: Optional[pd.DataFrame] = None
        self.obstacles = None

    def load(self) -> None:
        """
        Populate ``self.run`` by reading from ``self.log_path``.

        Implement log file parsing logic to populate self.run.
        """
        
        # Read drone data
        p = Path(self.log_path / "drones.csv")
        df = pd.read_csv(p)
        expected_columns = {"step", "drone_id", "x", "y", "z"}
        if not expected_columns.issubset(set(df.columns)):
            raise ValueError(
                f"Log file is missing required columns: {expected_columns}"
            )
        self.data = df

        # Read obstacle data
        obs_path = Path(self.log_path / "obstacles.json")
        if obs_path.exists():
            with open(obs_path, 'r') as f:
                self.obstacles = json.load(f).get("obstacles", [])

    def render(self) -> None:
        """
        Render the currently loaded run in a 3D view
        """
        if self.data is None:
            raise RuntimeError("No data loaded. Please call load() before render().")

        steps = sorted(self.data["step"].unique()) # List of step ids

        plotter = pv.Plotter()
        floor = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=20, j_size=20)
        plotter.add_mesh(floor, color="lightgray", style="wireframe", opacity=0.5)
        plotter.add_axes() # type: ignore

        # Initialize point cloud (collection of 3d points) with first step
        first_step = self.data[self.data["step"] == steps[0]]
        points = first_step[["x", "y", "z"]].to_numpy().astype('float32')
        cloud = pv.PolyData(points)
        plotter.add_points(
            cloud, color="orange", point_size=12, render_points_as_spheres=True
        )
        # Initialize text actor
        text_actor = plotter.add_text(
            f"Step {steps[0]}", 
            font_size=14, 
            position="upper_left"
        )
        
        # Set camera
        plotter.camera_position = 'iso'
        plotter.reset_camera() # type: ignore

        # Add obstacles to the scene
        if self.obstacles:
            for obstacle in self.obstacles:
            # Create a box mesh for each obstacle
                box = pv.Cube(
                    center=(
                    obstacle["posx"],
                    obstacle["posy"],
                    obstacle["posz"],
                    ),
                    x_length=obstacle["width"],
                    y_length=obstacle["depth"],
                    z_length=obstacle["height"],
                )
                plotter.add_mesh(box, color="red", opacity=0.5)
        
        # Callback function to update point cloud for each step -> easier to manage state + pausing
            # Also there is an issue with normal update
        # Use a list to maintain mutable states (to modify inside closuer)
        step_idx = [1] 
        
        def update_frame(step_count: int) -> None:
            """Callback function to update the animation that gets called repeatedly by the timer"""
            # No more steps to process
            if step_idx[0] >= len(steps):
                return 
            
            # Get current step
            step = steps[step_idx[0]]
            assert self.data is not None # Can do this because already checked for data outside
            step_df = self.data[self.data["step"] == step]
            points = step_df[["x", "y", "z"]].to_numpy().astype('float32')
            
            # Update elements
            cloud.points = points
            text_actor.SetText(2, f"Step {step}")
            
            # Move to next step
            step_idx[0] += 1
            
            if step_idx[0] < len(steps):
                plotter.add_timer_event(
                    max_steps=1,
                    duration=1000,
                    callback=update_frame
                ) # type: ignore
        
        # Starting timer for animation (to move on to step 1)
        plotter.add_timer_event(
            max_steps=1,
            duration=1000,
            callback=update_frame
        ) # type: ignore
        
        plotter.show() # Start event loop of window


# Testing
"""if __name__ == "__main__":
    here = Path(__file__).parent
    path = here / "test_log_path"
    vis = OfflineVisualizer(log_path=path)
    vis.load()
    vis.render()
"""