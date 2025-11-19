"""
Provides offline visualization capabilities for simulation playback and analysis.
This module will load simulation logs from disk and render 3D visualizations of drone trajectories for post-simulation analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import json

try:
    import pyvista as pv  # type: ignore[import-untyped]
except ModuleNotFoundError:
    print("WARNING: Rendering requires pyvista, which is not installed.")
    pv = None  # type: ignore[assignment]


class OfflineVisualizer:
    """
    Loads simulation logs from disk and produces offline visualizations.
    """

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        # self.run: Optional[SimulationRun] = None
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
            with open(obs_path, "r") as f:
                self.obstacles = json.load(f).get("obstacles", [])

    def render(self) -> None:
        """
        Render the currently loaded run in a 3D view
        """
        if pv is None:
            raise RuntimeError(
                "Rendering requires pyvista, which is not installed. "
                "Install pyvista to enable visualization support."
            )
        if self.data is None:
            raise RuntimeError("No data loaded. Please call load() before render().")

        steps = sorted(self.data["step"].unique())  # List of step ids

        plotter = pv.Plotter(window_size=[1920, 1080])
        floor = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=20, j_size=20)
        plotter.add_mesh(floor, color="lightgray", style="wireframe", opacity=0.5)
        plotter.add_axes()  # type: ignore

        # Initialize point cloud (collection of 3d points) with first step
        first_step = self.data[self.data["step"] == steps[0]]
        points = first_step[["x", "y", "z"]].to_numpy().astype("float32")
        cloud = pv.PolyData(points)
        plotter.add_points(
            cloud, color="orange", point_size=12, render_points_as_spheres=True
        )

        # Initialize text actor
        text_actor = plotter.add_text(
            f"Step {steps[0]}", font_size=14, position="upper_left"
        )

        # Set camera
        plotter.camera_position = "iso"
        plotter.reset_camera()  # type: ignore

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

        # Use a list to maintain mutable states (to modify inside closure)
        step_idx = [1]
        running = [False]

        def update_frame(step_count: int) -> None:
            """Callback function to update the animation that gets called repeatedly by the timer"""
            # Pause/Play logic
            if not running[0]:
                return
            # No more steps to process
            if step_idx[0] >= len(steps):
                running[0] = False
                return

            # Get current step and previous step
            step = steps[step_idx[0]]
            prev_step = steps[step_idx[0] - 1]

            assert self.data is not None
            step_df = self.data[self.data["step"] == step]
            prev_step_df = self.data[self.data["step"] == prev_step]
            points = step_df[["x", "y", "z"]].to_numpy().astype("float32")

            # Draw lines from previous positions to current positions for each drone
            for _, row in step_df.iterrows():
                drone_id = row["drone_id"]
                current_pos = np.array([row["x"], row["y"], row["z"]])

                # Find this drone's position in the previous step
                prev_row = prev_step_df[prev_step_df["drone_id"] == drone_id]

                if not prev_row.empty:
                    prev_pos = prev_row[["x", "y", "z"]].to_numpy()[0]

                    # Create line between previous and current position
                    line = pv.Line(prev_pos, current_pos)

                    # Add line and keep it persistent (don't remove)
                    plotter.add_mesh(line, color="blue", line_width=3)

            # Update elements
            cloud.points = points
            text_actor.SetText(2, f"Step {step}")

            # Move to next step
            step_idx[0] += 1

            if running[0] and step_idx[0] < len(steps):
                plotter.add_timer_event(
                    max_steps=1,
                    duration=250,  # Note -> if too big, it may create multiple timers and skip frames
                    callback=update_frame,
                )  # type: ignore

        def checkbox_callback(checked: bool) -> None:
            if checked:
                start_animation()
            else:
                stop_animation()

        def start_animation():
            # Start if not running and steps remain
            if not running[0]:
                running[0] = True
                if step_idx[0] < len(steps):
                    plotter.add_timer_event(
                        max_steps=1, duration=0, callback=update_frame
                    )  # type: ignore

        def stop_animation():
            running[0] = False

        plotter.add_checkbox_button_widget(
            checkbox_callback,
            value=False,
            position=(10, 10),
            size=60,
            color_on="green",
            color_off="red",
        )

        plotter.show()


if __name__ == "__main__":
    log_path = Path(__file__).resolve().parents[2] / "test_log_path"
    vis = OfflineVisualizer(log_path)
    vis.load()
    vis.render()
