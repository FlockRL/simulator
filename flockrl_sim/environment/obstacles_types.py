from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

Bounds = Tuple[float, float, float, float, float, float]  # (x_min, x_max, y_min, y_max, z_min, z_max)

class OBBMixin:
    def _rotation_matrix(self, orientation):
        roll, pitch, yaw = orientation or (0, 0, 0)

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll),  np.cos(roll)]
        ])
        Ry = np.array([
            [ np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0, 0, 1]
        ])
        return Rz @ Ry @ Rx

    def _obb_intersect(self, origin, direction, max_distance, half_sizes):
        pos = np.array(self.position)
        R = self._rotation_matrix(self.orientation)
        Rt = R.T

        o_local = Rt @ (origin - pos)
        d_local = Rt @ direction
        
        with np.errstate(divide='ignore', invalid='ignore'):
            t1 = (-half_sizes - o_local) / d_local
            t2 = ( half_sizes - o_local) / d_local


        tmin = np.maximum.reduce(np.minimum(t1, t2))
        tmax = np.minimum.reduce(np.maximum(t1, t2))

        if tmax < 0 or tmin > tmax or tmin > max_distance:
            return None

        t_hit = tmin if tmin >= 0 else tmax
        if t_hit > max_distance:
            return None

        p_local = o_local + t_hit * d_local

        eps = 1e-6
        nx = np.array([1,0,0])
        ny = np.array([0,1,0])
        nz = np.array([0,0,1])

        normal_local = np.zeros(3)
        if abs(p_local[0] - half_sizes[0]) < eps: normal_local = nx
        elif abs(p_local[0] + half_sizes[0]) < eps: normal_local = -nx
        elif abs(p_local[1] - half_sizes[1]) < eps: normal_local = ny
        elif abs(p_local[1] + half_sizes[1]) < eps: normal_local = -ny
        elif abs(p_local[2] - half_sizes[2]) < eps: normal_local = nz
        elif abs(p_local[2] + half_sizes[2]) < eps: normal_local = -nz

        hit_point = pos + R @ p_local
        normal = R @ normal_local

        return float(t_hit), hit_point, normal

@dataclass
class Obstacle:
    id: str
    type: str  # e.g., "wall", "gate", "sphere", "box", "pyramid"
    position: Tuple[float, float, float]  # (x, y, z)
    orientation: Optional[Tuple[float, float, float]]  # Euler angles (roll, pitch, yaw) in radians

    def ray_intersect(
        self,
        origin: np.ndarray,      # shape=(3,)
        direction: np.ndarray,   # shape=(3,), normalized
        max_distance: float
    ) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
        """
        Returns: (distance, hit_point, normal) or None if no hit
        """
        return None

@dataclass
class Wall(Obstacle, OBBMixin):
    length: float
    height: float
    thickness: float
    gate_ids: Tuple[str, ...] = ()

    def linked_gate_ids(self) -> Tuple[str, ...]:
        """Return all gate IDs associated with this wall."""
        return self.gate_ids

    def ray_intersect(
        self,
        origin: np.ndarray,      # shape=(3,)
        direction: np.ndarray,   # shape=(3,), normalized
        max_distance: float
    ) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
        half = np.array([self.length/2, self.thickness/2, self.height/2])
        return self._obb_intersect(origin, direction, max_distance, half)

@dataclass
class Gate(Obstacle):
    width: float
    height: float
    thickness: float

    def ray_intersect(
        self,
        origin: np.ndarray,      # shape=(3,)
        direction: np.ndarray,   # shape=(3,), normalized
        max_distance: float
    ) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
        half = np.array([self.width/2, self.frame_thickness/2, self.height/2])
        return self._obb_intersect(origin, direction, max_distance, half)

@dataclass
class Clutter(Obstacle):
    subtype: str  # e.g. "rectangular_prism"

    def ray_intersect(
        self,
        origin: np.ndarray,      # shape=(3,)
        direction: np.ndarray,   # shape=(3,), normalized
        max_distance: float
    ) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
        """
        Returns: (distance, hit_point, normal) or None if no hit
        """
        return None


@dataclass
class RectangularPrism(Clutter, OBBMixin):
    length: float
    width: float
    height: float

    def ray_intersect(
        self,
        origin: np.ndarray,      # shape=(3,)
        direction: np.ndarray,   # shape=(3,), normalized
        max_distance: float
    ) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
        half = np.array([self.length/2, self.width/2, self.height/2])
        return self._obb_intersect(origin, direction, max_distance, half)