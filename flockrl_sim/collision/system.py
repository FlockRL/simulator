from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np

from ..environment import Environment
from ..environment.obstacles_types import RectangularPrism
from ..geometry import OBB, sphere_intersect_obb, point_in_obb
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

    """

    environment: Environment
    drone_radius: float
    restitution: float

    def __call__(self, state: SwarmState) -> tuple[SwarmState, dict]:
        """
        Detect collisions and compute collision responses.

        Returns:
            (state, info) where:
            - state is the (unmodified) SwarmState passed in
            - info is a dict with:
                    "collisions": List[CollisionInfo]
        """
        # Degenerate case:
        if state.pos is None or state.vel is None or state.ids is None:
            return state, {"collisions": []}

        collisions: List[CollisionInfo] = []

        bounds = self.environment.bounds
        obstacles = self.environment.obstacles

        # Run collision checks

        # 1. Bounds collisions
        if bounds is not None:
            collisions.extend(self.check_bounds_collision(state, bounds))

        # 2. Wall + clutter collisions
        if obstacles:
            collisions.extend(self.check_wall_collision(state, obstacles))
            collisions.extend(self.check_clutter_collision(state, obstacles))

        info = {"collisions": collisions}
        return state, info

    def check_bounds_collision(
        self, state: SwarmState, bounds: Any
    ) -> List[CollisionInfo]:
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
            if x - r < x_min:
                faces.append(
                    (
                        np.array([1.0, 0.0, 0.0]),
                        np.array([x_min, y, z]),
                        (x_min + r - x),
                    )
                )
            if x + r > x_max:
                faces.append(
                    (
                        np.array([-1.0, 0.0, 0.0]),
                        np.array([x_max, y, z]),
                        (x - (x_max - r)),
                    )
                )
            if y - r < y_min:
                faces.append(
                    (
                        np.array([0.0, 1.0, 0.0]),
                        np.array([x, y_min, z]),
                        (y_min + r - y),
                    )
                )
            if y + r > y_max:
                faces.append(
                    (
                        np.array([0.0, -1.0, 0.0]),
                        np.array([x, y_max, z]),
                        (y - (y_max - r)),
                    )
                )
            if z - r < z_min:
                faces.append(
                    (
                        np.array([0.0, 0.0, 1.0]),
                        np.array([x, y, z_min]),
                        (z_min + r - z),
                    )
                )
            if z + r > z_max:
                faces.append(
                    (
                        np.array([0.0, 0.0, -1.0]),
                        np.array([x, y, z_max]),
                        (z - (z_max - r)),
                    )
                )

            for normal, face_point, pen in faces:
                contact_point = np.array(
                    [
                        np.clip(x, x_min, x_max),
                        np.clip(y, y_min, y_max),
                        np.clip(z, z_min, z_max),
                    ]
                )
                new_pos = pos + pen * normal
                rebound_vel = self.apply_rebound(
                    state.vel[i], normal, restitution=self.restitution
                )
                collisions.append(
                    CollisionInfo(
                        drone_id=drone_id,
                        collision_type="bounds",
                        normal_vector=normal,
                        contact_point=contact_point,
                        penetration_depth=float(abs(pen)),
                        rebound_velocity=rebound_vel,
                        new_position=new_pos,
                    )
                )

        return collisions

    def check_wall_collision(
        self, state: SwarmState, obstacles: List[Any]
    ) -> List[CollisionInfo]:
        """
        Check for collisions with static walls.

        Implements gate pass-through logic: if a drone is inside a gate volume
        that is embedded in a wall, the wall collision is ignored for that drone.
        """
        collisions = []

        walls = [obs for obs in obstacles if obs.type == "wall"]
        gates = [obs for obs in obstacles if obs.type == "gate"]

        gate_map = {gate.id: gate for gate in gates}

        for i, pos in enumerate(state.pos):
            drone_id = state.ids[i]
            drone_vel = state.vel[i]

            for wall in walls:
                # Check if drone is inside any of this wall's gates and skip collision with this wall if true
                if self._is_drone_inside_any_gate(pos, wall, gate_map):
                    continue

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
                    collision_type="wall",
                )

                if collision_info is not None:
                    collisions.append(collision_info)

        return collisions

    def check_clutter_collision(
        self, state: SwarmState, obstacles: List[Any]
    ) -> List[CollisionInfo]:
        """
        Check for collisions with clutter objects (rectangular prisms, spheres, etc.).

        Detects collisions between drones and obstacle objects, computing contact points,
        penetration depths, and rebound velocities for each collision.
        """
        collisions: List[CollisionInfo] = []
        prisms = [obs for obs in obstacles if isinstance(obs, RectangularPrism)]
        r = self.drone_radius

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
                    collision_type="clutter",
                )

                if collision_info is not None:
                    collisions.append(collision_info)

        # Sphere collisions
        spheres = [
            obs
            for obs in obstacles
            if obs.type == "sphere" or hasattr(obs, "radius")
        ]

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
                    rebound_vel = self.apply_rebound(
                        drone_vel, normal, restitution=self.restitution
                    )

                    collisions.append(
                        CollisionInfo(
                            drone_id=drone_id,
                            collision_type="sphere",
                            normal_vector=normal.astype(float),
                            contact_point=contact_point.astype(float),
                            penetration_depth=float(penetration),
                            rebound_velocity=rebound_vel.astype(float),
                            new_position=new_pos.astype(float),
                        )
                    )

        return collisions

    def check_rectangular_prism_collision(
        self,
        drone_pos: np.ndarray,
        drone_vel: np.ndarray,
        drone_id: int,
        box_position: np.ndarray,
        box_dimensions: tuple[float, float, float],  # (length, width, height)
        box_orientation: tuple[float, float, float],
        collision_type: str,
    ) -> CollisionInfo | None:
        """
        Check collision between a single drone and a single rectangular prism.

        Uses AABB collision detection for axis-aligned boxes (orientation = [0,0,0]),
        and OBB collision detection for rotated boxes.

        Args:
            drone_pos: Drone position, shape (3,)
            drone_vel: Drone velocity, shape (3,)
            drone_id: ID of the drone
            box_position: Center position of the box, shape (3,)
            box_dimensions: (length, width, height) of the box
            box_orientation: Euler angles (roll, pitch, yaw) in radians
            collision_type: Type string for CollisionInfo ("wall" or "clutter")

        Returns:
            CollisionInfo if collision detected, None otherwise
        """
        # Use AABB collision detection if orientation is zero (axis-aligned)
        if np.allclose(box_orientation, [0.0, 0.0, 0.0], atol=1e-9):
            return self._check_aabb_collision(
                drone_pos=drone_pos,
                drone_vel=drone_vel,
                drone_id=drone_id,
                box_position=box_position,
                box_dimensions=box_dimensions,
                collision_type=collision_type,
            )

        # Use OBB collision detection for all other cases
        return self._check_obb_collision(
            drone_pos=drone_pos,
            drone_vel=drone_vel,
            drone_id=drone_id,
            box_position=box_position,
            box_dimensions=box_dimensions,
            box_orientation=box_orientation,
            collision_type=collision_type,
        )

    def _check_aabb_collision(
        self,
        drone_pos: np.ndarray,
        drone_vel: np.ndarray,
        drone_id: int,
        box_position: np.ndarray,
        box_dimensions: tuple[float, float, float],
        collision_type: str,
    ) -> CollisionInfo | None:
        """
        Check collision between a drone and an axis-aligned bounding box (AABB).

        This is the original implementation for axis-aligned boxes (no rotation).
        Deprecated: Use _check_obb_collision instead, which handles both AABB and OBB.
        This function is kept for reference or future AABB-only optimizations.
        """
        r = self.drone_radius
        length, width, height = box_dimensions
        half_extents = np.array([length * 0.5, width * 0.5, height * 0.5], dtype=float)
        box_center = np.array(box_position, dtype=float)

        # Transform drone position to box's local coordinate system (which is just world space for AABB)
        pos_local = drone_pos - box_center

        # Find closest point on the box (in local space) to the drone
        closest_local = np.clip(pos_local, -half_extents, half_extents)

        # Vector from closest point to drone center (in local space)
        diff_local = pos_local - closest_local
        dist_sq = float(np.dot(diff_local, diff_local))

        # Check if drone sphere intersects the box
        if dist_sq <= r * r:
            dist = np.sqrt(dist_sq)

            if dist > 1e-12:
                # Normal case: drone center is outside the box
                normal_local = diff_local / dist
                contact_local = closest_local
                penetration = r - dist
            else:
                # Special case: drone center is inside the box
                # Find the closest face and push out along that axis
                dx_min = half_extents[0] + pos_local[0]  # Distance to -X face
                dx_max = half_extents[0] - pos_local[0]  # Distance to +X face
                dy_min = half_extents[1] + pos_local[1]  # Distance to -Y face
                dy_max = half_extents[1] - pos_local[1]  # Distance to +Y face
                dz_min = half_extents[2] + pos_local[2]  # Distance to -Z face
                dz_max = half_extents[2] - pos_local[2]  # Distance to +Z face

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

                face_dist, normal_local, contact_local = min(
                    candidates, key=lambda t: t[0]
                )
                penetration = r + face_dist

            # For AABB, local space is the same as world space (no rotation)
            normal_world = normal_local
            contact_world = box_center + contact_local
            new_pos = drone_pos + penetration * normal_world

            # Compute rebound velocity
            rebound_vel = self.apply_rebound(
                drone_vel, normal_world, restitution=self.restitution
            )

            return CollisionInfo(
                drone_id=drone_id,
                collision_type=collision_type,
                normal_vector=normal_world.astype(float),
                contact_point=contact_world.astype(float),
                penetration_depth=float(penetration),
                rebound_velocity=rebound_vel.astype(float),
                new_position=new_pos.astype(float),
            )

        # No collision
        return None

    def _check_obb_collision(
        self,
        drone_pos: np.ndarray,
        drone_vel: np.ndarray,
        drone_id: int,
        box_position: np.ndarray,
        box_dimensions: tuple[float, float, float],
        box_orientation: tuple[float, float, float] | None,
        collision_type: str,
    ) -> CollisionInfo | None:
        """
        Check collision between a drone and an oriented bounding box (OBB).

        Delegates to geometry module.
        """
        length, width, height = box_dimensions
        half_extents = np.array([length * 0.5, width * 0.5, height * 0.5], dtype=float)

        obb = OBB(
            center=np.array(box_position, dtype=float),
            half_extents=half_extents,
            orientation=box_orientation,
        )

        result = sphere_intersect_obb(drone_pos, self.drone_radius, obb)

        if result is None:
            return None

        penetration, contact_point, normal = result

        new_pos = drone_pos + penetration * normal
        rebound_vel = self.apply_rebound(
            drone_vel, normal, restitution=self.restitution
        )

        return CollisionInfo(
            drone_id=drone_id,
            collision_type=collision_type,
            normal_vector=normal.astype(float),
            contact_point=contact_point.astype(float),
            penetration_depth=float(penetration),
            rebound_velocity=rebound_vel.astype(float),
            new_position=new_pos.astype(float),
        )

    def apply_rebound(
        self, velocity: np.ndarray, normal: np.ndarray, restitution: float
    ) -> np.ndarray:
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

    def _is_drone_inside_any_gate(
        self, drone_pos: np.ndarray, wall: Any, gate_map: dict
    ) -> bool:
        """
        Check if a drone is inside any of the gates embedded in a wall.

        Args:
            drone_pos: Drone position, shape (3,)
            wall: Wall obstacle with gate_ids attribute
            gate_map: Dictionary mapping gate IDs to gate objects

        Returns:
            True if drone is inside any gate volume, False otherwise
        """
        gate_ids = getattr(wall, "gate_ids", ())
        if not gate_ids:
            return False

        for gate_id in gate_ids:
            gate = gate_map[gate_id]
            if self._is_point_inside_gate(drone_pos, gate):
                return True

        return False

    def _is_point_inside_gate(self, point: np.ndarray, gate: Any) -> bool:
        """
        Check if a point is inside a gate's bounding volume.

        Gates are rectangular volumes defined by (width, thickness, height).
        For axis-aligned gates, this is a simple AABB test.

        Args:
            point: 3D point to test, shape (3,)
            gate: Gate obstacle with position, width, height, thickness, orientation

        Returns:
            True if point is inside gate volume, False otherwise
        """
        gate_pos = np.array(gate.position, dtype=float)

        # Gate dimensions: (width, thickness, height) map to (x, y, z) half-extents
        half_extents = np.array(
            [gate.width * 0.5, gate.thickness * 0.5, gate.height * 0.5], dtype=float
        )

        obb = OBB(
            center=gate_pos, half_extents=half_extents, orientation=gate.orientation
        )

        return point_in_obb(point, obb)
