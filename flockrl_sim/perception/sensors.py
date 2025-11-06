"""
Perception subsystem scaffolding.

This module defines the interfaces the perception team will implement to expose
sensor data (ray casts, neighbor queries, etc.) to learning agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..environment.obstacles import Environment
from ..state import SwarmState


@dataclass
class SensorConfig:
    """
    Configuration for a perception sensor suite.

    Fields:
        max_range: Maximum sensing distance (meters).
        num_rays: Number of ray-cast beams in the virtual LIDAR.
        max_neighbour_range: Maximum distance to consider a drone as a neighbor (meters).
    """

    max_range: float = 50.0
    num_rays: int = 128
    max_neighbour_range: float = 10.0

@dataclass
class SensorReading:
    """
    Container for per-drone sensor outputs.

    Fields:
        ranges: Array of ray-cast distances, shape=(num_rays,).
        hits: Boolean array indicating whether each ray hit an obstacle.
        neighbor_vectors: Optional array of relative vectors to nearby drones.
        metadata: Arbitrary extra data (surface normals, obstacle IDs, etc.).
    """

    ranges: np.ndarray
    hits: np.ndarray
    neighbor_vectors: Optional[np.ndarray] = None
    metadata: Dict[str, np.ndarray] = field(default_factory=dict)


class PerceptionSystem:
    """
    Generates observation features for each drone using ray casting.
    """

    def __init__(self, environment: Environment, config: Optional[SensorConfig] = None) -> None:
        self.environment = environment
        self.config = config or SensorConfig()

    def reset(self, config: Optional[SensorConfig] = None) -> None:
        """
        Reset the perception system with an optional updated sensor configuration.
        """
        pass

    def observe(self, state: SwarmState) -> List[SensorReading]:
        """
        Compute sensor readings for every drone in state.

        Returns:
            List of SensorReading instances maintaining the same ordering as found in state
            (i.e., readings[i] corresponds to state.pos[i]).
        """
        pass
