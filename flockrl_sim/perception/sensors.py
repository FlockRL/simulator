"""
Perception subsystem scaffolding.

This module defines the interfaces the perception team will implement to expose
sensor data (ray casts, neighbor queries, etc.) to learning agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import logging

from ..environment.obstacles import Environment
from ..environment.obstacles_types import Gate, RectangularPrism, Wall
from ..geometry import (
    OBB,
    build_rotation_matrix,
    points_in_obb_batch,
    ray_intersect_obb_batch,
)
from ..state import SwarmState

logger = logging.getLogger(__name__)

@dataclass
class SensorConfig:
    """
    Configuration for a perception sensor suite.

    Fields:
        max_range: Maximum sensing distance (meters)
        num_rays: Number of ray-cast beams in the virtual LIDAR
        max_neighbour_range: Maximum distance to consider a drone as a neighbor (meters)
    """

    max_range: float
    num_rays: int
    max_neighbour_range: float


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
        config: SensorConfig,
        seed: Optional[int] = None,
    ) -> None:
        self.environment = environment
        self.config = config
        self.rays = generate_rays(self.config.num_rays, seed)
        self._prepare_obstacle_cache()

    def reset(
        self, config: SensorConfig, seed: Optional[int] = None
    ) -> None:
        """
        Reset the perception system with an optional updated sensor configuration.
        """

        self.config = config
        self.rays = generate_rays(self.config.num_rays, seed)

    def _prepare_obstacle_cache(self) -> None:
        """Pre-compute obstacle OBB data for efficient batch raycasting."""
        self._ray_obstacles: List[Dict] = []
        self._gate_cache: Dict[str, Dict] = {}

        for obs in self.environment.obstacles:
            center = np.array(obs.position, dtype=float)
            R = build_rotation_matrix(obs.orientation)

            if isinstance(obs, Gate):
                half = np.array(
                    [obs.width * 0.5, obs.thickness * 0.5, obs.height * 0.5]
                )
                self._gate_cache[obs.id] = {
                    "center": center,
                    "half_extents": half,
                    "rotation": R,
                }
                continue  # gates are transparent to rays

            if isinstance(obs, Wall):
                half = np.array(
                    [obs.length * 0.5, obs.thickness * 0.5, obs.height * 0.5]
                )
                gate_ids = obs.gate_ids
            elif isinstance(obs, RectangularPrism):
                half = np.array(
                    [obs.length * 0.5, obs.width * 0.5, obs.height * 0.5]
                )
                gate_ids = ()
            else:
                logger.warning(
                    "Obstacle type %r has no OBB geometry and will be ignored by the perception system.",
                    type(obs).__name__,
                )
                continue

            self._ray_obstacles.append(
                {
                    "center": center,
                    "half_extents": half,
                    "rotation": R,
                    "type": obs.type,
                    "gate_ids": gate_ids,
                }
            )

    def observe(self, state: SwarmState) -> List[SensorReading]:
        """
        Compute sensor readings for every drone in state.

        Uses vectorized batch raycasting: all N*M rays are tested against each
        obstacle in a single numpy operation instead of per-ray Python loops.

        Args:
            state: Current state of the swarm

        Returns:
            List of SensorReading instances maintaining the same ordering as found in state
            (i.e., readings[i] corresponds to state.pos[i])
        """

        N = state.pos.shape[0]
        M = self.config.num_rays
        max_range = self.config.max_range

        # Build all ray origins and directions for batch processing
        # Each drone casts M rays, total K = N*M rays
        origins = np.repeat(state.pos, M, axis=0)  # (K, 3)
        directions = np.tile(self.rays, (N, 1))  # (K, 3)

        # Initialize all distances to max range
        distances = np.full(N * M, max_range)

        # Test rays against each obstacle in one batch operation per obstacle
        for obs_data in self._ray_obstacles:
            obs_dists = ray_intersect_obb_batch(
                origins,
                directions,
                obs_data["center"],
                obs_data["half_extents"],
                obs_data["rotation"],
                max_range,
            )

            # Filter wall hits that pass through gates
            if obs_data["type"] == "wall" and obs_data["gate_ids"]:
                hit_mask = obs_dists < max_range
                if np.any(hit_mask):
                    hit_points = (
                        origins[hit_mask]
                        + obs_dists[hit_mask, np.newaxis] * directions[hit_mask]
                    )

                    in_any_gate = np.zeros(hit_mask.sum(), dtype=bool)
                    for gate_id in obs_data["gate_ids"]:
                        gate = self._gate_cache[gate_id]
                        in_any_gate |= points_in_obb_batch(
                            hit_points,
                            gate["center"],
                            gate["half_extents"],
                            gate["rotation"],
                        )

                    # Invalidate hits inside gates
                    hit_indices = np.where(hit_mask)[0]
                    obs_dists[hit_indices[in_any_gate]] = max_range

            # Update closest distances
            np.minimum(distances, obs_dists, out=distances)

        # Reshape to per-drone arrays
        all_dists = distances.reshape(N, M)
        all_hits = all_dists < max_range

        # Neighbor detection (already vectorized)
        neighbor_pos = state.pos[None, :, :] - state.pos[:, None, :]
        vel = state.vel if state.vel is not None else np.zeros_like(state.pos)
        neighbor_vel = vel[None, :, :] - vel[:, None, :]
        neighbor_dist = np.linalg.norm(neighbor_pos, axis=-1)
        np.fill_diagonal(neighbor_dist, float("inf"))

        # Build per-drone SensorReadings
        readings = []
        for i in range(N):
            neighbor_vectors = np.zeros((0, 6), dtype=float)
            if self.config.max_neighbour_range > 0:
                in_range = neighbor_dist[i] < self.config.max_neighbour_range
                neighbor_indices = np.where(in_range)[0]
                if neighbor_indices.size:
                    order = np.argsort(neighbor_dist[i, neighbor_indices])
                    ordered_indices = neighbor_indices[order]
                    neighbor_vectors = np.concatenate(
                        (neighbor_pos[i, ordered_indices], neighbor_vel[i, ordered_indices]),
                        axis=1,
                    )

            readings.append(SensorReading(all_dists[i], all_hits[i], neighbor_vectors))

        return readings
