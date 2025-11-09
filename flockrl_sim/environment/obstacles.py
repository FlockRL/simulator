"""Environment and obstacle data containers.

Obstacles team: Define specific obstacle types (Wall, Gate, Sphere, Box, etc.) as separate classes or subclasses based on your design doc.

IMPORTANT: Coordinate with Collision team on obstacle geometry representation needed for collision detection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum, auto
import random

from flockrl_sim.environment.obstacles_types import Obstacle

Bounds = Tuple[float, float, float, float, float, float]  # (x_min, x_max, y_min, y_max, z_min, z_max)



@dataclass
class Environment:
    bounds: Bounds = (-5.0, 5.0, -5.0, 5.0, 0.0, 5.0)
    obstacles: List[Obstacle] = field(default_factory=list)
    seed: Optional[int] = None

    def set_bounds(self, bounds: Bounds) -> None:
        self.bounds = bounds
        print(f"Environment bounds set to {self.bounds}")


    def add_obstacle(self, obstacle: Obstacle) -> None:
        self.obstacles.append(obstacle)
        print(f"Added {obstacle}")

    def get_obstacle_by_id(self, obstacle_id: str) -> Optional[Obstacle]:
        for obs in self.obstacles:
            if obs.id == obstacle_id:
                print(f"Found obstacle: {obs}")
                return obs
        return None

    def summary(self) -> str:
        print("Generating environment summary...")
        print(f"Bounds: {self.bounds}")
        print(f"Seed: {self.seed}")
        print(f"Number of obstacles: {len(self.obstacles)}")
        return (
            f"Environment bounds: {self.bounds}\n"
            f"Seed: {self.seed}\n"
            f"Number of obstacles: {len(self.obstacles)}"
        )


class EnvironmentBuilder:
    def __init__(self, config: Optional[Environment] = None, bounds: Optional[Bounds] = None) -> None:
        self.config = config or Environment()
        if bounds is not None:
            self.config.set_bounds(bounds)


    def add_random_obstacles(self, n: int = 5) -> "EnvironmentBuilder":
        if self.config.seed is not None:
            random.seed(self.config.seed)

        for i in range(n):
            x = random.uniform(self.config.bounds[0], self.config.bounds[1])
            y = random.uniform(self.config.bounds[2], self.config.bounds[3])
            z = random.uniform(self.config.bounds[4], self.config.bounds[5])
            obstacle = Obstacle(id=str(i), type="wall", position=(x, y, z))
            self.config.add_obstacle(obstacle)
        return self

    def build(self) -> Environment:
        return self.config