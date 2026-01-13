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
from ..geometry import OBB, point_in_obb
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

    def reset(
        self, config: SensorConfig, seed: Optional[int] = None
    ) -> None:
        """
        Reset the perception system with an optional updated sensor configuration.
        """

        self.config = config
        self.rays = generate_rays(self.config.num_rays, seed)

    def _is_point_inside_gate(self, point: np.ndarray, gate) -> bool:
        """
        Check if a point is inside a gate's bounding volume, this is to filter out rays that hit a portion of the wall which contains a gate
        """
        gate_pos = np.array(gate.position, dtype=float)

        # Gate dimensions: (width, thickness, height) map to (x, y, z) half-extents
        half_extents = np.array(
            [gate.width * 0.5, gate.thickness * 0.5, gate.height * 0.5], dtype=float
        )

        obb = OBB(
            center=gate_pos, half_extents=half_extents, orientation=gate.orientation
        )

        return point_in_obb(point, obb)

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

        # Build gate map for filtering wall hits that pass through gates
        gates = [obs for obs in self.environment.obstacles if obs.type == "gate"]
        gate_map = {gate.id: gate for gate in gates}

        # for each drone, calculate relative position/velocity of other drones in the swarm
        neighbor_pos = state.pos[None, :, :] - state.pos[:, None, :]
        vel = state.vel if state.vel is not None else np.zeros_like(state.pos)
        neighbor_vel = vel[None, :, :] - vel[:, None, :]

        # for each drone, calculate relative distance of other drones in the swarm
        neighbor_dist = np.linalg.norm(neighbor_pos, axis=-1)

        # drones will not consider itself as a neighbor
        np.fill_diagonal(neighbor_dist, float("inf"))

        readings = []
        # for each drone get a SensorReading
        for i in range(N):
            # by default there is no ray hit and ray-cast distance is maximum sensing distance
            ray_dists = np.full(M, self.config.max_range)
            ray_hits = np.full(M, False)

            # for each ray, get its ray-cast distance and whether it hit an obstacle
            for j in range(M):
                # Get ray intersections from all obstacles, tracking which obstacle each hit came from
                raycast_results = [
                    (obst, obst.ray_intersect(
                        state.pos[i], self.rays[j], self.config.max_range
                    ))
                    for obst in self.environment.obstacles
                ]

                # Filter out None results
                hits = [(obst, res) for obst, res in raycast_results if res is not None]

                # Filter out wall hits that pass through gates
                filtered_hits = []
                for obst, hit_info in hits:
                    # Check if this is a wall hit
                    if obst.type == "wall":
                        _, hit_point, _ = hit_info
                        # Check if hit point is inside any of this wall's gates
                        gate_ids = obst.gate_ids
                        is_in_gate = False
                        for gate_id in gate_ids:
                            gate = gate_map[gate_id]
                            if self._is_point_inside_gate(hit_point, gate):
                                is_in_gate = True
                                break
                        # Skip this wall hit if it's inside a gate (ray passes through)
                        if is_in_gate:
                            continue

                    # Also filter out gate hits (gates should be transparent to rays)
                    if obst.type == "gate":
                        continue

                    filtered_hits.append(hit_info)

                if filtered_hits:
                    ray_dists[j] = min(hit[0] for hit in filtered_hits)
                    ray_hits[j] = True

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

            readings.append(SensorReading(ray_dists, ray_hits, neighbor_vectors))

        return readings
