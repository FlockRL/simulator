"""
Ray casting utilities for LIDAR and sensor simulation.

Implement ray-obstacle intersection tests to simulate distance sensors.
Key algorithms needed: ray-plane, ray-sphere, ray-AABB (axis-aligned bounding box).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from ..environment.obstacles import Obstacle


RayHit = Tuple[float, np.ndarray, np.ndarray]
"""Ray casting result: (distance, hit_point, normal)"""


def raycast(
    origin: np.ndarray,
    direction: np.ndarray,
    obstacles: list[Obstacle],
    max_distance: float,
) -> Optional[RayHit]:
    """
    Cast a ray and return closest hit within max_distance.
    
    For each obstacle, compute ray intersection and return the nearest one.
    Standard computer graphics ray tracing problem.
    
    Args:
        origin: Ray origin (shape=(3,))
        direction: Normalized direction vector (shape=(3,))
        obstacles: List of obstacles to test against
        max_distance: Maximum ray distance [meters]
    
    Returns:
        (distance, hit_point, normal) or None if no hit
    """
    pass


def raycast_batch(
    origins: np.ndarray,
    directions: np.ndarray,
    obstacles: list[Obstacle],
    max_distance: float,
) -> np.ndarray:
    """
    Cast multiple rays at once
    
    Consider performance when implementing this.
    
    Args:
        origins: Ray origins (shape=(N, 3))
        directions: Normalized direction vectors (shape=(N, 3))
        obstacles: List of obstacles to test against
        max_distance: Maximum ray distance [meters]
    
    Returns:
        Distances to closest hit for each ray (shape=(N,))
    """
    pass
