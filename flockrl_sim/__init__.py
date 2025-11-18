"""FlockRL Simulator - Public API exports."""

from .collision.system import CollisionInfo, CollisionSystem
from .config import (
    CollisionConfig,
    EnvironmentConfig,
    SimulationConfig,
    VisualizationConfig,
)
from .environment.obstacles import Environment, EnvironmentBuilder, Obstacle
from .perception import PerceptionSystem, RayHit, SensorConfig, SensorReading
from .simulator import CoreSimulator, SimulationFrame, SimulationRun
from .state import SwarmState
try:
    from .visualization.renderer import OfflineVisualizer
except ModuleNotFoundError as exc:  # pragma: no cover - exercised when optional deps missing
    class OfflineVisualizer:  # type: ignore[override]
        """Placeholder that explains missing optional visualization dependency."""

        def __init__(self, *_args, **_kwargs):
            raise ModuleNotFoundError(
                "OfflineVisualizer requires optional visualization dependencies (e.g. pyvista). "
                "Install them to enable visualization support."
            ) from exc

__all__ = [
    # Core state
    "SwarmState",
    # Simulator and logging
    "CoreSimulator",
    "SimulationFrame",
    "SimulationRun",
    # Environment
    "Environment",
    "EnvironmentBuilder",
    "Obstacle",
    # Collision
    "CollisionSystem",
    "CollisionInfo",
    # Perception
    "PerceptionSystem",
    "SensorConfig",
    "SensorReading",
    "RayHit",
    # Visualization
    "OfflineVisualizer",
    # Configs
    "SimulationConfig",
    "EnvironmentConfig",
    "CollisionConfig",
    "VisualizationConfig",
]
