"""
Offline visualization entry point for simulation logs saved by CoreSimulator.save_run().
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import json


class OfflineVisualizer:
    """
    Load a JSON log produced by CoreSimulator.save_run() and render it with
    a configurable backend (plotly or pyvista).
    """

    def __init__(
        self,
        log_path: Path,
        render_mode: str = "plotly",
        playback_speed: float = 250.0,
    ) -> None:
        self.log_path = Path(log_path)
        self.render_mode = render_mode
        self.playback_speed = playback_speed
        self.frames: List[Dict[str, Any]] = []
        self.metadata: Dict[str, Any] = {}
        self.obstacles: List[Dict[str, Any]] = []

    def load(self) -> None:
        """
        Load simulation run from JSON file and extract obstacles.
        """
        if not self.log_path.exists():
            raise FileNotFoundError(f"Log file not found: {self.log_path}")

        with open(self.log_path, "r") as f:
            data = json.load(f)

        self.metadata = data["metadata"]
        self.frames = data["frames"]

        if not self.frames:
            raise ValueError("No frames found in log file")

        self.obstacles = self._extract_obstacles(self.metadata)

    def render(
        self,
        mode: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Render the loaded simulation run using the selected backend.
        """
        if not self.frames:
            raise RuntimeError("No data loaded. Please call load() before render().")

        selected_mode = (mode or self.render_mode).lower()
        if selected_mode in {"plotly", "plotty", "dash"}:
            from .plotly_renderer import PlotlyRenderer

            renderer = PlotlyRenderer(
                frames=self.frames,
                metadata=self.metadata,
                obstacles=self.obstacles,
                playback_speed=self.playback_speed,
            )
            renderer.render(**kwargs)
            return
        if selected_mode in {"pyvista", "pv"}:
            from .pyvista_renderer import PyvistaRenderer

            renderer = PyvistaRenderer(
                frames=self.frames,
                metadata=self.metadata,
                obstacles=self.obstacles,
                playback_speed=self.playback_speed,
            )
            renderer.render(**kwargs)
            return

        raise ValueError(
            f"Unknown render mode '{selected_mode}'. Use 'plotly' or 'pyvista'."
        )

    @staticmethod
    def _extract_obstacles(metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        if "obstacles" in metadata:
            obstacles = metadata["obstacles"]
            if isinstance(obstacles, list):
                return obstacles

        if "environment" not in metadata:
            return []
        environment = metadata["environment"]
        if isinstance(environment, dict):
            if "obstacles" in environment:
                env_obstacles = environment["obstacles"]
                if isinstance(env_obstacles, list):
                    return env_obstacles

        return []


if __name__ == "__main__":
    log_path = Path(__file__).resolve().parents[2] / "demo_simulation_output.json"
    vis = OfflineVisualizer(log_path)
    vis.load()
    vis.render()
