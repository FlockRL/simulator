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
        max_range: Maximum sensing distance (meters)
        num_rays: Number of ray-cast beams in the virtual LIDAR
        max_neighbour_range: Maximum distance to consider a drone as a neighbor (meters)
    """

    max_range: float = 50.0
    num_rays: int = 128
    max_neighbour_range: float = 10.0

@dataclass
class SensorReading:
    """
    Container for per-drone sensor outputs.

    Fields:
        ranges: Array of ray-cast distances, shape (M,)
        hits: Boolean array indicating whether each ray hit an obstacle
        neighbor_vectors: Array of relative vectors to nearby drones
        metadata: Arbitrary extra data (surface normals, obstacle IDs, etc.)
    """

    ranges: np.ndarray
    hits: np.ndarray
    neighbor_vectors: np.ndarray
    metadata: Dict[str, np.ndarray] = field(default_factory=dict)


def generate_rays(num_rays: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generates normalized ray direction vectors that are uniformly distributed on a unit sphere.

    Args:
        num_rays: Number of ray-cast beams in the virtual LIDAR
        seed: A optional seed to initialize the random number generator

    Returns:
        Spherically uniform unit vectors, shape (M, 3)
    """

    # initialize random number generator
    rng = np.random.default_rng(seed)

    # sample spherically uniform unit vector, following the method from https://angms.science/doc/RM/randUnitVec.pdf
    rays = rng.standard_normal((num_rays, 3))
    norm = np.sum(rays**2, axis=1, keepdims=True) ** 0.5
    rays /= norm

    return rays


class PerceptionSystem:
    """
    Generates observation features for each drone using ray casting.
    """

    def __init__(
        self,
        environment: Environment,
        config: Optional[SensorConfig] = None,
        seed: Optional[int] = None,
    ) -> None:

        self.environment = environment
        self.config = config or SensorConfig()
        self.rays = generate_rays(self.config.num_rays, seed)

    def reset(
        self, config: Optional[SensorConfig] = None, seed: Optional[int] = None
    ) -> None:
        """
        Reset the perception system with an optional updated sensor configuration.
        """

        self.config = config or SensorConfig()
        self.rays = generate_rays(self.config.num_rays, seed)

    def observe(self, state: SwarmState) -> List[SensorReading]:
        """
        Compute sensor readings for every drone in state.

        Args:
            state: Current state of the swarm

        Returns:
            List of SensorReading instances maintaining the same ordering as found in state
            (i.e., readings[i] corresponds to state.pos[i])
        """

        N = state.pos.shape[0]
        M = self.config.num_rays

        # for each drone, calculate relative position/velocity of other drones in the swarm
        neighbor_pos = state.pos[None, :, :] - state.pos[:, None, :]
        vel = state.vel if state.vel is not None else np.zeros_like(state.pos)
        neighbor_vel = vel[None, :, :] - vel[:, None, :]

        # for each drone, calculate relative distance of other drones in the swarm
        neighbor_dist = np.linalg.norm(neighbor_pos, axis=-1)

        # drones will not consider itself as a neighbor
        np.fill_diagonal(neighbor_dist, float("inf"))

        # get mask to select relative position/velocity of drones that are considered neighbors
        mask = neighbor_dist < self.config.max_neighbour_range

        readings = []
        # for each drone get a SensorReading
        for i in range(N):
            # by default there is no ray hit and ray-cast distance is maximum sensing distance
            ray_dists = np.full(M, self.config.max_range)
            ray_hits = np.full(M, False)

            # for each ray, get its ray-cast distance and whether it hit an obstacle
            for j in range(M):
                raycast_results = [
                    obst.ray_intersect(
                        state.pos[i], self.rays[j], self.config.max_range
                    )
                    for obst in self.environment.obstacles
                ]

                hits = [res for res in raycast_results if res is not None]
                if hits:
                    ray_dists[j] = min(hit[0] for hit in hits)
                    ray_hits[j] = True

            readings.append(
                SensorReading(
                    ray_dists,
                    ray_hits,
                    np.concatenate(
                        (neighbor_pos[i, mask[i]], neighbor_vel[i, mask[i]]), axis=1
                    ),
                )
            )

        return readings
