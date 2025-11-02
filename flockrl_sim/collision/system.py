from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np

from ..environment import Environment
from ..state import SwarmState


@dataclass
class CollisionInfo:
    """
    Will be generated for each collision that occurs.

    Data fields:
        drone_id: ID of the drone that collided
        collision_type: Assuming only drones can collide into objects,
            this will be the type of object collided into ("wall", "clutter")
        normal_vector: Surface normal at collision point, shape (3,)
        contact_point: Point of contact in world coordinates, shape (3,)
        penetration_depth: How far the drone penetrated the obstacle [m]
        rebound_velocity: Velocity of the drone after the collision, shape (3,)
        new_position: The new position of the drone after the collision, shape (3,)
    """
    drone_id: int
    collision_type: str
    normal_vector: np.ndarray  # shape (3,)
    contact_point: np.ndarray  # shape (3,)
    penetration_depth: float
    rebound_velocity: np.ndarray  # shape (3,)
    new_position: np.ndarray  # shape (3,)

@dataclass
class CollisionSystem:
    """
    Collision system that will handle collision detection and response for drone swarms in the simulation environment.
    
    Boilerplate class for now, feel free to change as much as you want.
    """

    environment: Environment
    drone_radius: float = 1.0  # Hardcoded value for drone radius (can change)

    def __call__(self, state: SwarmState) -> tuple[SwarmState, dict]:
        """
        Detect and resolve collisions, return updated state and info dict.
        
        The info dict should include a "collisions" key containing a list of
        CollisionInfo objects for logging and visualization.
        
        Collision team: Implement the full collision pipeline here.
        """
        pass

    def check_bounds_collision(self, state: SwarmState, bounds: Any) -> List[CollisionInfo]:
        """
        Check for collisions with environment bounds.
        """

        x_min, y_min, z_min, x_max, y_max, z_max = bounds
        collisions = []

        for i, pos in enumerate(state.pos):
            drone_id = state.ids[i]
            x, y, z = pos

            if (x - self.drone_radius < x_min or x + self.drone_radius > x_max or
                y - self.drone_radius < y_min or y + self.drone_radius > y_max or
                z - self.drone_radius < z_min or z + self.drone_radius > z_max):
                collisions.append(CollisionInfo(
                    drone_id=drone_id,
                    collision_type="bounds",
                    details={"position": pos}
                ))

        return collisions

    def check_wall_collision(self, state: SwarmState, obstacles: List[Any]) -> List[CollisionInfo]:
        """
        Check for collisions with static walls.
        """
        collisions = []

        walls = [obs for obs in obstacles if obs.type == "wall"]
        
        for i, pos in enumerate(state.pos):
            drone_id = state.ids[i]
            drone_vel = state.vel[i]
            
            for wall in walls:
                if hasattr(wall, 'point') and hasattr(wall, 'normal'):
                    dist = np.dot(pos - wall.point, wall.normal)
                    
                    if abs(dist) < self.drone_radius:
                        contact_point = pos - dist * wall.normal
                        
                        rebound_vel = self.apply_rebound(drone_vel, wall.normal, restitution=0.8)
                        
                        new_pos = pos + (self.drone_radius - dist) * wall.normal
                        
                        collisions.append(CollisionInfo(
                            drone_id=drone_id,
                            collision_type="wall",
                            normal_vector=wall.normal,
                            contact_point=contact_point,
                            penetration_depth=self.drone_radius - abs(dist),
                            rebound_velocity=rebound_vel,
                            new_position=new_pos
                        ))
        
        return collisions

    def check_clutter_collision(self, state: SwarmState, obstacles: List[Any]) -> List[CollisionInfo]:
        """
        Check for collisions with clutter objects (spheres, boxes, etc.).

        Unneeded for now, but can be used for future physics simulation
        """
        pass

    def apply_rebound(self, velocity: np.ndarray, normal: np.ndarray, restitution: float) -> np.ndarray:
        """
        Compute reflected velocity after collision.
        
        Args:
            velocity: Current velocity vector, shape (3,)
            normal: Surface normal at collision point, shape (3,)
            restitution: Coefficient of restitution (1.0 = elastic, 0.0 = inelastic)
        
        Returns:
            Updated velocity vector after rebound, shape (3,)
        """
        pass
