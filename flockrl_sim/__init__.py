"""FlockRL Simulator - Public API exports."""

from .collision.system import CollisionInfo, CollisionSystem
from .gym_env import FlockRLGymEnv, load_config
from .gym_logging import EpisodeLogger, EpisodeResult, TrajectoryData
from .rewards import RewardFunction
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
    # Config
    "load_config",
    # Gymnasium
    "FlockRLGymEnv",
    # Gymnasium logging
    "EpisodeLogger",
    "EpisodeResult",
    "TrajectoryData",
    # Reward functions
    "RewardFunction",
]
