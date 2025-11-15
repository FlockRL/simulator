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

        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        collisions: List[CollisionInfo] = []
        r = self.drone_radius

        for i, pos in enumerate(state.pos):
            drone_id = state.ids[i]
            x, y, z = pos

            faces = []
            if x - r < x_min: faces.append( (np.array([ 1.0, 0.0, 0.0]), np.array([x_min, y, z]), (x_min + r - x)) )
            if x + r > x_max: faces.append( (np.array([-1.0, 0.0, 0.0]), np.array([x_max, y, z]), (x - (x_max - r))) )
            if y - r < y_min: faces.append( (np.array([0.0,  1.0, 0.0]), np.array([x, y_min, z]), (y_min + r - y)) )
            if y + r > y_max: faces.append( (np.array([0.0, -1.0, 0.0]), np.array([x, y_max, z]), (y - (y_max - r))) )
            if z - r < z_min: faces.append( (np.array([0.0, 0.0,  1.0]), np.array([x, y, z_min]), (z_min + r - z)) )
            if z + r > z_max: faces.append( (np.array([0.0, 0.0, -1.0]), np.array([x, y, z_max]), (z - (z_max - r))) )

            for normal, face_point, pen in faces:
                contact_point = np.array([np.clip(x, x_min, x_max),
                                        np.clip(y, y_min, y_max),
                                        np.clip(z, z_min, z_max)])
                new_pos = pos + pen * normal
                rebound_vel = self.apply_rebound(state.vel[i], normal, restitution=0.8)
                collisions.append(CollisionInfo(
                    drone_id=drone_id,
                    collision_type="bounds",
                    normal_vector=normal,
                    contact_point=contact_point,
                    penetration_depth=float(abs(pen)),
                    rebound_velocity=rebound_vel,
                    new_position=new_pos
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
                # Extract dimensions: Wall has (length, height, thickness)
                # Map to (length, width, height) for the prism method
                box_dims = (wall.length, wall.thickness, wall.height)
                box_pos = np.array(wall.position, dtype=float)
                box_orientation = wall.orientation

                collision_info = self.check_rectangular_prism_collision(
                    drone_pos=pos,
                    drone_vel=drone_vel,
                    drone_id=drone_id,
                    box_position=box_pos,
                    box_dimensions=box_dims,
                    box_orientation=box_orientation,
                    collision_type="wall"
                )

                if collision_info is not None:
                    collisions.append(collision_info)

        return collisions

    def check_clutter_collision(self, state: SwarmState, obstacles: List[Any]) -> List[CollisionInfo]:
        """
        Check for collisions with clutter objects (spheres, boxes, etc.).

        Unneeded for now, but can be used for future physics simulation
        """
        collisions: List[CollisionInfo] = []
        prisms = [obs for obs in obstacles if getattr(obs, "type", None) == "RectangularPrism"]

        for i, pos in enumerate(state.pos):
            drone_id = state.ids[i]
            drone_vel = state.vel[i]

            for prism in prisms:
                # Extract dimensions: RectangularPrism has (length, width, height)
                box_dims = (prism.length, prism.width, prism.height)
                box_pos = np.array(prism.position, dtype=float)
                box_orientation = prism.orientation

                collision_info = self.check_rectangular_prism_collision(
                    drone_pos=pos,
                    drone_vel=drone_vel,
                    drone_id=drone_id,
                    box_position=box_pos,
                    box_dimensions=box_dims,
                    box_orientation=box_orientation,
                    collision_type="clutter"
                )

                if collision_info is not None:
                    collisions.append(collision_info)

        # Sphere collisions
        spheres = [obs for obs in obstacles if getattr(obs, "type", None) == "sphere" or hasattr(obs, "radius")]

        for i, pos in enumerate(state.pos):
            drone_id = state.ids[i]
            drone_vel = state.vel[i]

            for sphere in spheres:
                center = np.array(sphere.position, dtype=float)
                sphere_r = float(getattr(sphere, "radius", 0.0))

                diff = pos - center
                dist_sq = float(np.dot(diff, diff))
                cutoff = (r + sphere_r) ** 2

                if dist_sq < cutoff:
                    dist = np.sqrt(dist_sq)

                    if dist > 1e-12:
                        normal = diff / dist
                    else:
                        # Drone center coincides with sphere center; pick an arbitrary normal to push drone out along +x.
                        normal = np.array([1.0, 0.0, 0.0], dtype=float)

                    contact_point = center + sphere_r * normal
                    penetration = (r + sphere_r) - dist
                    new_pos = pos + penetration * normal
                    rebound_vel = self.apply_rebound(drone_vel, normal, restitution=0.8)

                    collisions.append(CollisionInfo(
                        drone_id=drone_id,
                        collision_type="sphere",
                        normal_vector=normal.astype(float),
                        contact_point=contact_point.astype(float),
                        penetration_depth=float(penetration),
                        rebound_velocity=rebound_vel.astype(float),
                        new_position=new_pos.astype(float),
                    ))

        return collisions

    def check_rectangular_prism_collision(
        self,
        drone_pos: np.ndarray,
        drone_vel: np.ndarray,
        drone_id: int,
        box_position: np.ndarray,
        box_dimensions: tuple[float, float, float],  # (length, width, height)
        box_orientation: tuple[float, float, float] | None,
        collision_type: str
    ) -> CollisionInfo | None:
        """
        Check collision between a single drone and a single rectangular prism.

        Args:
            drone_pos: Drone position, shape (3,)
            drone_vel: Drone velocity, shape (3,)
            drone_id: ID of the drone
            box_position: Center position of the box, shape (3,)
            box_dimensions: (length, width, height) of the box
            box_orientation: Euler angles (roll, pitch, yaw) in radians, or None for AABB
            collision_type: Type string for CollisionInfo ("wall" or "clutter")

        Returns:
            CollisionInfo if collision detected, None otherwise
        """
        # Check if orientation is axis-aligned (only AABB supported for now)
        if box_orientation is not None:
            if not np.allclose(box_orientation, [0.0, 0.0, 0.0], atol=1e-9):
                raise NotImplementedError(
                    f"Oriented bounding boxes (OBB) are not yet supported. "
                    f"Box has orientation {box_orientation}, but only axis-aligned boxes "
                    f"(orientation = None or [0, 0, 0]) are currently implemented."
                )

        r = self.drone_radius
        pos = drone_pos
        length, width, height = box_dimensions
        half = np.array([length * 0.5, width * 0.5, height * 0.5], dtype=float)
        center = np.array(box_position, dtype=float)
        pmin = center - half
        pmax = center + half

        closest = np.maximum(pmin, np.minimum(pos, pmax))
        diff = pos - closest
        dist_sq = float(np.dot(diff, diff))

        if dist_sq < r * r:
            dist = np.sqrt(dist_sq)

            if dist > 1e-12:
                normal = diff / dist
                contact_point = closest
                penetration = r - dist
                new_pos = pos + penetration * normal
            else:
                dx_min = (pos[0] - pmin[0])
                dx_max = (pmax[0] - pos[0])
                dy_min = (pos[1] - pmin[1])
                dy_max = (pmax[1] - pos[1])
                dz_min = (pos[2] - pmin[2])
                dz_max = (pmax[2] - pos[2])

                candidates = [
                    (dx_min, np.array([-1.0,  0.0,  0.0]), np.array([pmin[0], pos[1], pos[2]])),
                    (dx_max, np.array([ 1.0,  0.0,  0.0]), np.array([pmax[0], pos[1], pos[2]])),
                    (dy_min, np.array([ 0.0, -1.0,  0.0]), np.array([pos[0], pmin[1], pos[2]])),
                    (dy_max, np.array([ 0.0,  1.0,  0.0]), np.array([pos[0], pmax[1], pos[2]])),
                    (dz_min, np.array([ 0.0,  0.0, -1.0]), np.array([pos[0], pos[1], pmin[2]])),
                    (dz_max, np.array([ 0.0,  0.0,  1.0]), np.array([pos[0], pos[1], pmax[2]])),
                ]
                face_dist, normal, contact_point = min(candidates, key=lambda t: t[0])

                penetration = r
                new_pos = pos + penetration * normal

            rebound_vel = self.apply_rebound(drone_vel, normal, restitution=0.8)

            return CollisionInfo(
                drone_id=drone_id,
                collision_type=collision_type,
                normal_vector=normal.astype(float),
                contact_point=contact_point.astype(float),
                penetration_depth=float(penetration),
                rebound_velocity=rebound_vel.astype(float),
                new_position=new_pos.astype(float),
            )

        # No collision
        return None

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
        n = normal / (np.linalg.norm(normal) + 1e-12)
        v_n = np.dot(velocity, n) * n
        v_t = velocity - v_n
        return v_t - restitution * v_n
