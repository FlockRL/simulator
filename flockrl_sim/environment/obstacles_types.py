from dataclasses import dataclass
from typing import Tuple, Optional, Union, List


@dataclass
class Obstacle:
    """
    Obstacle definition.
    
    TODO - Obstacles team: Expand this class with fields needed for your
    obstacle types (position, size, geometry, orientation, etc.).
    Consider creating specific subclasses for different obstacle types
    (Wall, Gate, Sphere, Box, Pyramid) as described in your design doc.
    """
    id: str
    type: str  # e.g., "wall", "gate", "sphere", "box", "pyramid"
    position: Tuple[float, float, float]  # (x, y, z)
    orientation: Optional[Tuple[float, float, float]] = None


@dataclass
class Wall(Obstacle):
    length: float
    height: float
    thickness: float
    gate_id: Optional[str] = None  # ID of gate if wall has a gate

@dataclass
class Gate(Obstacle):
    width: float
    height: float
    frame_thickness: float

@dataclass
class Clutter(Obstacle):
    subtype: str  # e.g. "box", "pyramid", "sphere"

