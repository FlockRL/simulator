"""
Geometric primitives and intersection tests for the simulation.

Handles mathematical operations for:
- Rotation matrices
- OBB (Oriented Bounding Box) intersections
- Ray casting against geometric shapes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np


@dataclass
class OBB:
    """Oriented Bounding Box."""

    center: np.ndarray  # (3,)
    half_extents: np.ndarray  # (3,)
    orientation: Optional[Tuple[float, float, float]] = (
        None  # (roll, pitch, yaw) in radians
    )

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Get the 3x3 rotation matrix for this OBB."""
        return build_rotation_matrix(self.orientation)


def build_rotation_matrix(
    orientation: Optional[Tuple[float, float, float]],
) -> np.ndarray:
    """
    Build a rotation matrix from Euler angles (roll, pitch, yaw).

    Args:
        orientation: Euler angles (roll, pitch, yaw) in radians, or None for identity

    Returns:
        3x3 rotation matrix
    """
    if orientation is None:
        return np.eye(3, dtype=float)

    roll, pitch, yaw = orientation

    # Check if this is effectively an identity rotation
    if np.allclose([roll, pitch, yaw], [0.0, 0.0, 0.0], atol=1e-9):
        return np.eye(3, dtype=float)

    # Build rotation matrices for each axis
    Rx = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]],
        dtype=float,
    )

    Ry = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ],
        dtype=float,
    )

    Rz = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]],
        dtype=float,
    )

    # Combined rotation: Rz * Ry * Rx
    return Rz @ Ry @ Rx


def ray_intersect_obb(
    ray_origin: np.ndarray, ray_direction: np.ndarray, obb: OBB, max_distance: float
) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    """
    Compute intersection between a ray and an OBB.

    Args:
        ray_origin: Ray start point (3,)
        ray_direction: Normalized ray direction (3,)
        obb: The Oriented Bounding Box
        max_distance: Maximum ray length

    Returns:
        (distance, hit_point, normal) or None if no intersection
    """
    pos = obb.center
    R = obb.rotation_matrix
    Rt = R.T
    half_sizes = obb.half_extents

    # Transform ray to OBB's local coordinate system
    o_local = Rt @ (ray_origin - pos)
    d_local = Rt @ ray_direction

    # Slab method for AABB in local space
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (-half_sizes - o_local) / d_local
        t2 = (half_sizes - o_local) / d_local

    tmin = np.maximum.reduce(np.minimum(t1, t2))
    tmax = np.minimum.reduce(np.maximum(t1, t2))
    
    if __debug__:
        inside = (tmax >= 0) & (tmin <= tmax) & (tmin < 0)
        assert not np.any(inside), "Ray origin inside OBB - collision should have ended episode"

    if tmax < 0 or tmin > tmax or tmin > max_distance:
        return None

    t_hit = tmin if tmin >= 0 else tmax
    if t_hit > max_distance:
        return None

    # Calculate hit point in local space
    p_local = o_local + t_hit * d_local

    # Calculate normal in local space
    eps = 1e-6
    nx = np.array([1, 0, 0], dtype=float)
    ny = np.array([0, 1, 0], dtype=float)
    nz = np.array([0, 0, 1], dtype=float)

    normal_local = np.zeros(3, dtype=float)

    if abs(p_local[0] - half_sizes[0]) < eps:
        normal_local = nx
    elif abs(p_local[0] + half_sizes[0]) < eps:
        normal_local = -nx
    elif abs(p_local[1] - half_sizes[1]) < eps:
        normal_local = ny
    elif abs(p_local[1] + half_sizes[1]) < eps:
        normal_local = -ny
    elif abs(p_local[2] - half_sizes[2]) < eps:
        normal_local = nz
    elif abs(p_local[2] + half_sizes[2]) < eps:
        normal_local = -nz

    # Transform back to world space
    hit_point = pos + R @ p_local
    normal = R @ normal_local

    return float(t_hit), hit_point, normal


def sphere_intersect_obb(
    sphere_center: np.ndarray, sphere_radius: float, obb: OBB
) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
    """
    Compute intersection/penetration between a sphere and an OBB.

    Args:
        sphere_center: Center of sphere (3,)
        sphere_radius: Radius of sphere
        obb: The Oriented Bounding Box

    Returns:
        (penetration_depth, contact_point_world, normal_world) or None
        Normal points OUT of the box (towards the sphere center).
    """
    R = obb.rotation_matrix

    # Transform sphere center to box's local coordinate system
    pos_local = R.T @ (sphere_center - obb.center)

    # Find closest point on the box (in local space) to the sphere center
    closest_local = np.clip(pos_local, -obb.half_extents, obb.half_extents)

    # Vector from closest point to sphere center (in local space)
    diff_local = pos_local - closest_local
    dist_sq = float(np.dot(diff_local, diff_local))

    # Check if sphere intersects the box
    if dist_sq > sphere_radius * sphere_radius:
        return None

    dist = np.sqrt(dist_sq)

    if dist > 1e-12:
        # Normal case: sphere center is outside the box
        # Normal points from box to sphere
        normal_local = diff_local / dist
        contact_local = closest_local
        penetration = sphere_radius - dist
    else:
        # Special case: sphere center is inside the box
        # Find the closest face and push out along that axis
        normal_local, contact_local, penetration = _compute_internal_collision(
            pos_local, obb.half_extents, sphere_radius
        )

    # Transform normal and contact point back to world space
    normal_world = R @ normal_local
    contact_world = obb.center + R @ contact_local

    return penetration, contact_world, normal_world


import numba as nb

@nb.njit(fastmath=True, cache=True)
def ray_intersect_obb_batch(
    origins: np.ndarray,
    directions: np.ndarray,
    obb_center: np.ndarray,
    obb_half_extents: np.ndarray,
    obb_rotation: np.ndarray,
    max_distance: float,
) -> np.ndarray:
    """Numba-optimized batch ray-OBB intersection."""
    K = origins.shape[0]
    result = np.full(K, max_distance, dtype=np.float64)
    
    for i in range(K):
        # Transform ray to local space
        o_local_x = 0.0; o_local_y = 0.0; o_local_z = 0.0
        d_local_x = 0.0; d_local_y = 0.0; d_local_z = 0.0
        
        # Manual matrix multiplication for speed
        ox = origins[i, 0] - obb_center[0]
        oy = origins[i, 1] - obb_center[1]
        oz = origins[i, 2] - obb_center[2]
        dx = directions[i, 0]
        dy = directions[i, 1]
        dz = directions[i, 2]
        
        for j in range(3):
            o_local_x += ox * obb_rotation[0, 0] + oy * obb_rotation[1, 0] + oz * obb_rotation[2, 0]
            o_local_y += ox * obb_rotation[0, 1] + oy * obb_rotation[1, 1] + oz * obb_rotation[2, 1]
            o_local_z += ox * obb_rotation[0, 2] + oy * obb_rotation[1, 2] + oz * obb_rotation[2, 2]
            
            d_local_x += dx * obb_rotation[0, 0] + dy * obb_rotation[1, 0] + dz * obb_rotation[2, 0]
            d_local_y += dx * obb_rotation[0, 1] + dy * obb_rotation[1, 1] + dz * obb_rotation[2, 1]
            d_local_z += dx * obb_rotation[0, 2] + dy * obb_rotation[1, 2] + dz * obb_rotation[2, 2]

        # Slab method
        tmin = -1e9
        tmax = 1e9
        
        # X-axis
        if abs(d_local_x) > 1e-8:
            tx1 = (-obb_half_extents[0] - o_local_x) / d_local_x
            tx2 = (obb_half_extents[0] - o_local_x) / d_local_x
            tmin = max(tmin, min(tx1, tx2))
            tmax = min(tmax, max(tx1, tx2))
        elif abs(o_local_x) > obb_half_extents[0]:
            continue # Ray is parallel and outside

        # Y-axis
        if abs(d_local_y) > 1e-8:
            ty1 = (-obb_half_extents[1] - o_local_y) / d_local_y
            ty2 = (obb_half_extents[1] - o_local_y) / d_local_y
            tmin = max(tmin, min(ty1, ty2))
            tmax = min(tmax, max(ty1, ty2))
        elif abs(o_local_y) > obb_half_extents[1]:
            continue
            
        # Z-axis
        if abs(d_local_z) > 1e-8:
            tz1 = (-obb_half_extents[2] - o_local_z) / d_local_z
            tz2 = (obb_half_extents[2] - o_local_z) / d_local_z
            tmin = max(tmin, min(tz1, tz2))
            tmax = min(tmax, max(tz1, tz2))
        elif abs(o_local_z) > obb_half_extents[2]:
            continue

        if tmax >= 0 and tmin <= tmax:
            t_hit = tmin if tmin >= 0 else tmax
            if t_hit <= max_distance:
                result[i] = t_hit
                
    return result


def points_in_obb_batch(
    points: np.ndarray,
    obb_center: np.ndarray,
    obb_half_extents: np.ndarray,
    obb_rotation: np.ndarray,
) -> np.ndarray:
    """
    Batch check if points are inside an OBB.

    Args:
        points: Points to test, shape (K, 3)
        obb_center: OBB center, shape (3,)
        obb_half_extents: OBB half extents, shape (3,)
        obb_rotation: OBB rotation matrix, shape (3, 3)

    Returns:
        Boolean mask, shape (K,). True if point is inside OBB.
    """
    local_points = (points - obb_center) @ obb_rotation  # (K, 3)
    return np.all(np.abs(local_points) <= obb_half_extents, axis=1)


def _compute_internal_collision(
    pos_local: np.ndarray, half_extents: np.ndarray, sphere_radius: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Helper to compute collision when a point is strictly inside an AABB.
    Returns (normal_local, contact_local, penetration).
    """
    # Distance to each face (positive means inside)
    dx_min = half_extents[0] + pos_local[0]
    dx_max = half_extents[0] - pos_local[0]
    dy_min = half_extents[1] + pos_local[1]
    dy_max = half_extents[1] - pos_local[1]
    dz_min = half_extents[2] + pos_local[2]
    dz_max = half_extents[2] - pos_local[2]

    candidates = [
        (
            dx_min,
            np.array([-1.0, 0.0, 0.0]),
            np.array([-half_extents[0], pos_local[1], pos_local[2]]),
        ),
        (
            dx_max,
            np.array([1.0, 0.0, 0.0]),
            np.array([half_extents[0], pos_local[1], pos_local[2]]),
        ),
        (
            dy_min,
            np.array([0.0, -1.0, 0.0]),
            np.array([pos_local[0], -half_extents[1], pos_local[2]]),
        ),
        (
            dy_max,
            np.array([0.0, 1.0, 0.0]),
            np.array([pos_local[0], half_extents[1], pos_local[2]]),
        ),
        (
            dz_min,
            np.array([0.0, 0.0, -1.0]),
            np.array([pos_local[0], pos_local[1], -half_extents[2]]),
        ),
        (
            dz_max,
            np.array([0.0, 0.0, 1.0]),
            np.array([pos_local[0], pos_local[1], half_extents[2]]),
        ),
    ]

    # Find the closest face (minimum distance to any face)
    face_dist, normal_local, contact_local = min(candidates, key=lambda t: t[0])

    # Penetration depth includes the entire radius since center is inside
    penetration = sphere_radius + face_dist

    return normal_local, contact_local, penetration
