"""
Plotly/Dash renderer for simulation runs saved by CoreSimulator.save_run().
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np

try:
    import dash
    from dash import dcc, html, Input, Output, State
    import plotly.graph_objects as go
    DASH_AVAILABLE = True
except ModuleNotFoundError:
    print("WARNING: Rendering requires dash and plotly, which are not installed.")
    print("Install with: pip install dash plotly")
    DASH_AVAILABLE = False


class PlotlyRenderer:
    """
    Render a simulation run using Dash + Plotly.
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

    def render(self, host: str = "127.0.0.1", port: int = 8050, debug: bool = False) -> None:
        """
        Render the loaded simulation run in an interactive Dash web application.
        """
        if not DASH_AVAILABLE:
            raise RuntimeError(
                "Rendering requires dash and plotly, which are not installed. "
                "Install with: pip install dash plotly"
            )
        if not self.frames:
            raise RuntimeError("No frames provided for rendering.")

        app = dash.Dash(__name__)
        initial_fig = self._create_figure(frame_idx=0, show_trajectories_up_to=0)

        app.layout = html.Div(
            [
                html.H1("FlockRL Simulation Visualizer", style={"textAlign": "center"}),
                dcc.Graph(id="3d-graph", figure=initial_fig, style={"height": "70vh"}),
                html.Div(
                    [
                        html.Button(
                            "Play",
                            id="play-button",
                            n_clicks=0,
                            style={"fontSize": "16px", "margin": "10px"},
                        ),
                        html.Button(
                            "Reset",
                            id="reset-button",
                            n_clicks=0,
                            style={"fontSize": "16px", "margin": "10px"},
                        ),
                        html.Div(
                            [
                                html.Label("Frame:"),
                                dcc.Slider(
                                    id="frame-slider",
                                    min=0,
                                    max=len(self.frames) - 1,
                                    value=0,
                                    marks={
                                        i: str(i)
                                        for i in range(
                                            0,
                                            len(self.frames),
                                            max(1, len(self.frames) // 10),
                                        )
                                    },
                                    tooltip={"placement": "bottom", "always_visible": True},
                                ),
                            ],
                            style={"width": "80%", "margin": "20px auto"},
                        ),
                        html.Div(
                            [
                                html.Label("Playback Speed (ms):"),
                                dcc.Slider(
                                    id="speed-slider",
                                    min=50,
                                    max=1000,
                                    value=self.playback_speed,
                                    step=50,
                                    marks={
                                        50: "50",
                                        250: "250",
                                        500: "500",
                                        1000: "1000",
                                    },
                                    tooltip={"placement": "bottom", "always_visible": True},
                                ),
                            ],
                            style={"width": "40%", "margin": "20px auto"},
                        ),
                        html.Div(
                            id="info-display",
                            style={
                                "textAlign": "center",
                                "fontSize": "18px",
                                "margin": "10px",
                            },
                        ),
                    ],
                    style={"textAlign": "center"},
                ),
                dcc.Interval(
                    id="interval-component",
                    interval=self.playback_speed,
                    n_intervals=0,
                    disabled=True,
                ),
                dcc.Store(id="is-playing", data=False),
                dcc.Store(id="current-frame", data=0),
            ]
        )

        @app.callback(
            [
                Output("play-button", "children"),
                Output("is-playing", "data"),
                Output("interval-component", "disabled"),
            ],
            [Input("play-button", "n_clicks")],
            [State("is-playing", "data")],
        )
        def toggle_play(n_clicks, is_playing):
            if n_clicks > 0:
                is_playing = not is_playing
            return ("Pause" if is_playing else "Play", is_playing, not is_playing)

        @app.callback(
            [Output("current-frame", "data"), Output("frame-slider", "value")],
            [Input("reset-button", "n_clicks")],
        )
        def reset_animation(n_clicks):
            return (0, 0)

        @app.callback(
            Output("interval-component", "interval"), [Input("speed-slider", "value")]
        )
        def update_speed(speed):
            return speed

        @app.callback(
            [
                Output("3d-graph", "figure"),
                Output("info-display", "children"),
                Output("frame-slider", "value", allow_duplicate=True),
                Output("current-frame", "data", allow_duplicate=True),
            ],
            [Input("interval-component", "n_intervals"), Input("frame-slider", "value")],
            [State("is-playing", "data"), State("current-frame", "data")],
            prevent_initial_call=True,
        )
        def update_figure(n_intervals, slider_value, is_playing, current_frame):
            ctx = dash.callback_context

            if not ctx.triggered:
                frame_idx = 0
            else:
                trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
                if trigger_id == "frame-slider":
                    frame_idx = slider_value
                elif trigger_id == "interval-component" and is_playing:
                    frame_idx = current_frame + 1
                    if frame_idx >= len(self.frames):
                        frame_idx = 0
                else:
                    frame_idx = current_frame

            fig = self._create_figure(frame_idx, show_trajectories_up_to=frame_idx)

            frame_state = self.frames[frame_idx]["state"]
            frame_time = frame_state.get("t", 0.0)
            info_text = (
                f"Frame: {frame_idx} / {len(self.frames) - 1} | "
                f"Time: {frame_time:.3f}s | Drones: {len(frame_state['pos'])}"
            )

            return (fig, info_text, frame_idx, frame_idx)

        print(f"\n{'=' * 60}")
        print("Starting Dash visualization server...")
        print(f"Open your browser to: http://{host}:{port}")
        print("Press Ctrl+C to stop the server")
        print(f"{'=' * 60}\n")

        app.run(host=host, port=port, debug=debug)

    def _create_figure(self, frame_idx: int, show_trajectories_up_to: int) -> "go.Figure":
        fig = go.Figure()

        for obstacle in self.obstacles:
            obstacle_traces = self._create_obstacle_mesh(obstacle)
            for trace in obstacle_traces:
                fig.add_trace(trace)

        if show_trajectories_up_to > 0:
            for step in range(1, min(show_trajectories_up_to + 1, len(self.frames))):
                prev_frame = self.frames[step - 1]
                curr_frame = self.frames[step]

                prev_state = prev_frame["state"]
                curr_state = curr_frame["state"]

                prev_positions = np.array(prev_state["pos"])
                curr_positions = np.array(curr_state["pos"])
                prev_ids = np.array(
                    prev_state.get("ids", list(range(len(prev_positions))))
                )
                curr_ids = np.array(
                    curr_state.get("ids", list(range(len(curr_positions))))
                )

                for i, drone_id in enumerate(curr_ids):
                    if i < len(curr_positions):
                        prev_idx = np.where(prev_ids == drone_id)[0]
                        if len(prev_idx) > 0 and prev_idx[0] < len(prev_positions):
                            prev_pos = prev_positions[prev_idx[0]]
                            curr_pos = curr_positions[i]
                            fig.add_trace(
                                go.Scatter3d(
                                    x=[prev_pos[0], curr_pos[0]],
                                    y=[prev_pos[1], curr_pos[1]],
                                    z=[prev_pos[2], curr_pos[2]],
                                    mode="lines",
                                    line=dict(color="blue", width=4),
                                    showlegend=False,
                                    hoverinfo="skip",
                                )
                            )

        current_state = self.frames[frame_idx]["state"]
        goals = current_state.get("goals")
        if goals is not None and len(goals) > 0:
            goals_array = np.array(goals)
            drone_ids = current_state.get("ids", list(range(len(goals))))

            for i, (goal, drone_id) in enumerate(zip(goals_array, drone_ids)):
                goal_box_traces = self._create_box_mesh(
                    cx=goal[0],
                    cy=goal[1],
                    cz=goal[2],
                    dx=0.5,
                    dy=0.5,
                    dz=0.5,
                    name=f"Goal (Drone {drone_id})",
                )
                for trace in goal_box_traces:
                    trace.color = "green"
                    trace.opacity = 0.2
                    trace.showlegend = i == 0
                    if i == 0:
                        trace.name = "Goals"
                    fig.add_trace(trace)

        positions = np.array(current_state["pos"])
        drone_ids = current_state.get("ids", list(range(len(positions))))

        fig.add_trace(
            go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode="markers",
                marker=dict(size=10, color="orange", line=dict(color="black", width=2)),
                text=[f"Drone {drone_id}" for drone_id in drone_ids],
                hoverinfo="text",
                name="Drones",
            )
        )

        fig.update_layout(
            scene=dict(
                xaxis=dict(title="X", gridcolor="lightgray"),
                yaxis=dict(title="Y", gridcolor="lightgray"),
                zaxis=dict(title="Z", gridcolor="lightgray"),
                aspectmode="data",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            ),
            showlegend=True,
            hovermode="closest",
            margin=dict(l=0, r=0, b=0, t=0),
        )

        return fig

    def _create_obstacle_mesh(self, obstacle: Dict[str, Any]) -> List["go.Mesh3d"]:
        obs_type = obstacle.get("type", "").lower()
        position = obstacle.get("position", (0, 0, 0))

        if isinstance(position, (list, tuple)) and len(position) == 3:
            pos_x, pos_y, pos_z = position
        else:
            pos_x = obstacle.get("posx", 0)
            pos_y = obstacle.get("posy", 0)
            pos_z = obstacle.get("posz", 0)

        if obs_type == "wall":
            length = obstacle.get("length", 1.0)
            height = obstacle.get("height", 1.0)
            thickness = obstacle.get("thickness", 0.1)
            return self._create_box_mesh(
                pos_x, pos_y, pos_z, length, thickness, height, "Wall"
            )
        if obs_type == "gate":
            width = obstacle.get("width", 1.0)
            height = obstacle.get("height", 1.0)
            thickness = obstacle.get("thickness", 0.1)
            return self._create_box_mesh(
                pos_x, pos_y, pos_z, width, thickness, height, "Gate"
            )
        if obs_type in ["clutter", "rectangularprism", "rectangular_prism"]:
            length = obstacle.get("length", obstacle.get("width", 1.0))
            width = obstacle.get("width", obstacle.get("depth", 1.0))
            height = obstacle.get("height", 1.0)
            return self._create_box_mesh(
                pos_x, pos_y, pos_z, length, width, height, "Clutter"
            )

        width = obstacle.get("width", obstacle.get("length", 1.0))
        depth = obstacle.get("depth", obstacle.get("width", 1.0))
        height = obstacle.get("height", 1.0)
        return self._create_box_mesh(
            pos_x, pos_y, pos_z, width, depth, height, "Obstacle"
        )

    def _create_box_mesh(
        self, cx: float, cy: float, cz: float, dx: float, dy: float, dz: float, name: str
    ) -> List["go.Mesh3d"]:
        vertices = np.array(
            [
                [cx - dx / 2, cy - dy / 2, cz - dz / 2],
                [cx + dx / 2, cy - dy / 2, cz - dz / 2],
                [cx + dx / 2, cy + dy / 2, cz - dz / 2],
                [cx - dx / 2, cy + dy / 2, cz - dz / 2],
                [cx - dx / 2, cy - dy / 2, cz + dz / 2],
                [cx + dx / 2, cy - dy / 2, cz + dz / 2],
                [cx + dx / 2, cy + dy / 2, cz + dz / 2],
                [cx - dx / 2, cy + dy / 2, cz + dz / 2],
            ]
        )

        i = [0, 0, 4, 4, 0, 0, 2, 2, 0, 0, 1, 1]
        j = [1, 2, 5, 6, 1, 5, 3, 7, 3, 7, 2, 6]
        k = [2, 3, 6, 7, 5, 4, 7, 6, 7, 4, 6, 5]

        mesh = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=i,
            j=j,
            k=k,
            color="red",
            opacity=0.15,
            name=name,
            showlegend=False,
            hoverinfo="name",
            flatshading=True,
        )

        return [mesh]
