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
    gate_id: Optional[str] = None  # ID of first gate for backward compatibility
    gate_ids: Tuple[str, ...] = ()

    def linked_gate_ids(self) -> Tuple[str, ...]:
        """Return all gate IDs associated with this wall."""
        if self.gate_ids:
            if self.gate_id and self.gate_id not in self.gate_ids:
                return self.gate_ids + (self.gate_id,)
            return self.gate_ids
        return (self.gate_id,) if self.gate_id else tuple()

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
