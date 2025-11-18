from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

from ..geometry import OBB, ray_intersect_obb

Bounds = Tuple[float, float, float, float, float, float]  # (x_min, x_max, y_min, y_max, z_min, z_max)


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
class Wall(Obstacle):
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
        obb = OBB(
            center=np.array(self.position),
            half_extents=half,
            orientation=self.orientation
        )
        return ray_intersect_obb(origin, direction, obb, max_distance)

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
        half = np.array([self.width/2, self.thickness/2, self.height/2])
        obb = OBB(
            center=np.array(self.position),
            half_extents=half,
            orientation=self.orientation
        )
        return ray_intersect_obb(origin, direction, obb, max_distance)

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
class RectangularPrism(Clutter):
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
        obb = OBB(
            center=np.array(self.position),
            half_extents=half,
            orientation=self.orientation
        )
        return ray_intersect_obb(origin, direction, obb, max_distance)