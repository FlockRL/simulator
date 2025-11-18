from dataclasses import dataclass
from typing import Tuple, Optional

Bounds = Tuple[float, float, float, float, float, float]  # (x_min, x_max, y_min, y_max, z_min, z_max)

@dataclass
class Obstacle:
    id: str
    type: str  # e.g., "wall", "gate", "sphere", "box", "pyramid"
    position: Tuple[float, float, float]  # (x, y, z)
    orientation: Optional[Tuple[float, float, float]]  # Euler angles (roll, pitch, yaw) in radians

@dataclass
class Wall(Obstacle):
    length: float
    height: float
    thickness: float
    gate_id: Optional[str]  # ID of gate if wall has a gate

@dataclass
class Gate(Obstacle):
    width: float
    height: float
    thickness: float

@dataclass
class Clutter(Obstacle):
    subtype: str  # e.g. "rectangular_prism"

@dataclass
class RectangularPrism(Clutter):
    length: float
    width: float
    height: float
