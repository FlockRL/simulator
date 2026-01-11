"""
PyVista renderer for simulation runs saved by CoreSimulator.save_run().
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    import pyvista as pv  # type: ignore[import-untyped]
except ModuleNotFoundError:
    pv = None  # type: ignore[assignment]


class PyvistaRenderer:
    """
    Render a simulation run using PyVista.
    """

    def __init__(
        self,
        frames: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        obstacles: List[Dict[str, Any]],
        playback_speed: float = 250.0,
    ) -> None:
        self.frames = frames
        self.metadata = metadata
        self.obstacles = obstacles
        self.playback_speed = playback_speed

    def render(self, window_size: Tuple[int, int] = (1920, 1080)) -> None:
        if pv is None:
            raise RuntimeError(
                "Rendering requires pyvista, which is not installed. "
                "Install pyvista to enable visualization support."
            )
        if not self.frames:
            raise RuntimeError("No frames provided for rendering.")

        plotter = pv.Plotter(window_size=list(window_size))
        bounds = self._get_bounds()
        floor = self._build_floor(bounds)
        plotter.add_mesh(floor, color="lightgray", style="wireframe", opacity=0.5)
        plotter.add_axes()  # type: ignore

        first_state = self.frames[0]["state"]
        points = np.array(first_state["pos"], dtype="float32")
        cloud = pv.PolyData(points)
        plotter.add_points(
            cloud, color="orange", point_size=12, render_points_as_spheres=True
        )

        text_actor = plotter.add_text("Frame 0", font_size=14, position="upper_left")

        plotter.camera_position = "iso"
        plotter.reset_camera()  # type: ignore

        for obstacle in self.obstacles:
            self._add_obstacle(plotter, obstacle)

        goals = first_state.get("goals")
        if goals:
            self._add_goals(plotter, goals)

        frame_idx = [1]
        running = [False]

        def update_frame(step_count: int) -> None:
            if not running[0]:
                return
            if frame_idx[0] >= len(self.frames):
                running[0] = False
                return

            step = frame_idx[0]
            prev_step = frame_idx[0] - 1

            curr_state = self.frames[step]["state"]
            prev_state = self.frames[prev_step]["state"]

            curr_positions = np.array(curr_state["pos"], dtype="float32")
            prev_positions = np.array(prev_state["pos"], dtype="float32")
            curr_ids = np.array(
                curr_state.get("ids", list(range(len(curr_positions))))
            )
            prev_ids = np.array(
                prev_state.get("ids", list(range(len(prev_positions))))
            )

            for i, drone_id in enumerate(curr_ids):
                if i < len(curr_positions):
                    prev_idx = np.where(prev_ids == drone_id)[0]
                    if len(prev_idx) > 0 and prev_idx[0] < len(prev_positions):
                        prev_pos = prev_positions[prev_idx[0]]
                        curr_pos = curr_positions[i]
                        line = pv.Line(prev_pos, curr_pos)
                        plotter.add_mesh(line, color="blue", line_width=3)

            cloud.points = curr_positions
            text_actor.SetText(2, f"Frame {step}")

            frame_idx[0] += 1

            if running[0] and frame_idx[0] < len(self.frames):
                plotter.add_timer_event(
                    max_steps=1,
                    duration=int(self.playback_speed),
                    callback=update_frame,
                )  # type: ignore

        def checkbox_callback(checked: bool) -> None:
            if checked:
                start_animation()
            else:
                stop_animation()

        def start_animation() -> None:
            if not running[0]:
                running[0] = True
                if frame_idx[0] < len(self.frames):
                    plotter.add_timer_event(
                        max_steps=1, duration=0, callback=update_frame
                    )  # type: ignore

        def stop_animation() -> None:
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

    def _get_bounds(self) -> Optional[Tuple[float, float, float, float, float, float]]:
        environment = self.metadata.get("environment")
        if isinstance(environment, dict):
            bounds = environment.get("bounds")
            if isinstance(bounds, (list, tuple)) and len(bounds) == 6:
                return tuple(float(v) for v in bounds)
        return None

    def _build_floor(self, bounds: Optional[Tuple[float, float, float, float, float, float]]):
        if bounds is None:
            return pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=20, j_size=20)
        xmin, xmax, ymin, ymax, zmin, _ = bounds
        center = ((xmin + xmax) / 2.0, (ymin + ymax) / 2.0, zmin)
        i_size = max(1.0, xmax - xmin)
        j_size = max(1.0, ymax - ymin)
        return pv.Plane(center=center, direction=(0, 0, 1), i_size=i_size, j_size=j_size)

    def _add_goals(self, plotter: "pv.Plotter", goals: List[List[float]]) -> None:
        for goal in goals:
            if len(goal) != 3:
                continue
            box = pv.Cube(
                center=(goal[0], goal[1], goal[2]), x_length=0.5, y_length=0.5, z_length=0.5
            )
            plotter.add_mesh(box, color="green", opacity=0.2)

    def _add_obstacle(self, plotter: "pv.Plotter", obstacle: Dict[str, Any]) -> None:
        obs_type = obstacle.get("type", "").lower()
        position = obstacle.get("position", (0, 0, 0))

        if isinstance(position, (list, tuple)) and len(position) == 3:
            pos_x, pos_y, pos_z = position
        else:
            pos_x = obstacle.get("posx", 0)
            pos_y = obstacle.get("posy", 0)
            pos_z = obstacle.get("posz", 0)

        if obs_type == "wall":
            x_len = obstacle.get("length", 1.0)
            y_len = obstacle.get("thickness", 0.1)
            z_len = obstacle.get("height", 1.0)
        elif obs_type == "gate":
            x_len = obstacle.get("width", 1.0)
            y_len = obstacle.get("thickness", 0.1)
            z_len = obstacle.get("height", 1.0)
        elif obs_type in ["clutter", "rectangularprism", "rectangular_prism"]:
            x_len = obstacle.get("length", obstacle.get("width", 1.0))
            y_len = obstacle.get("width", obstacle.get("depth", 1.0))
            z_len = obstacle.get("height", 1.0)
        else:
            x_len = obstacle.get("width", obstacle.get("length", 1.0))
            y_len = obstacle.get("depth", obstacle.get("width", 1.0))
            z_len = obstacle.get("height", 1.0)

        box = pv.Cube(
            center=(pos_x, pos_y, pos_z), x_length=x_len, y_length=y_len, z_length=z_len
        )
        plotter.add_mesh(box, color="red", opacity=0.5)
