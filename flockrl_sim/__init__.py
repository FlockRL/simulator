"""FlockRL Simulator - Public API exports."""

from .collision.system import CollisionInfo, CollisionSystem
from .gym_env import FlockRLGymEnv, load_config
from .gym_logging import EpisodeLogger, EpisodeResult
from .rewards import RewardFunction
from .environment.obstacles import Environment, EnvironmentBuilder, Obstacle
from .perception import PerceptionSystem, RayHit, SensorConfig, SensorReading
from .simulator import CoreSimulator, SimulationFrame, SimulationRun
from .state import SwarmState

# Note: Visualization modules should be imported directly from flockrl_sim.visualization
# to avoid dependency issues (e.g., OfflineVisualizer uses optional pyvista/plotly backends)

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
    # Reward functions
    "RewardFunction",
]
