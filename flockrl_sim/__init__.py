"""FlockRL Simulator - Public API exports."""

from .collision.system import CollisionInfo, CollisionSystem
from .config import CollisionConfig, EnvironmentConfig, SimulationConfig, VisualizationConfig
from .environment.obstacles import Environment, EnvironmentBuilder, Obstacle
from .perception import PerceptionSystem, RayHit, SensorConfig, SensorReading
from .simulator import CoreSimulator, SimulationFrame, SimulationRun
from .state import SwarmState
from .visualization.renderer import OfflineVisualizer

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
