"""Environment and obstacle data containers.

Obstacles team: Define specific obstacle types (Wall, Gate, Sphere, Box, etc.) as separate classes or subclasses based on your design doc.

IMPORTANT: Coordinate with Collision team on obstacle geometry representation needed for collision detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

Bounds = Tuple[float, float, float, float, float, float]  # (x_min, x_max, y_min, y_max, z_min, z_max)


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

@dataclass
class Environment:
    """
    Container for the simulation environment configuration.
    
    bounds: (x_min, x_max, y_min, y_max, z_min, z_max) defining the 3D simulation box
    obstacles: List of all obstacles in the environment
    seed: Random seed for reproducible environment generation
    """
    bounds: Bounds = (-5.0, 5.0, -5.0, 5.0, 0.0, 5.0)
    obstacles: List[Obstacle] = field(default_factory=list)
    seed: Optional[int] = None


class EnvironmentBuilder:
    """
    Builder for the simulation environment.
    """

    def __init__(self, config: Environment) -> None:
        self.config = config

    def build(self) -> Environment:
        """
        Return an Environment instance after applying build steps.
        
        Obstacles team: Implement environment generation logic here.
        """
        pass
