"""
Mock obstacles for perception team development and testing.

These mock obstacles implement the ray_intersect() interface contract
so the perception team can develop and test ray casting independently
of the obstacles team's implementation.

Interface Contract:
    def ray_intersect(
        self, 
        origin: np.ndarray,      # shape=(3,)
        direction: np.ndarray,   # shape=(3,), normalized
        max_distance: float
    ) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
        Returns: (distance, hit_point, normal) or None if no hit
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np


def _to_array3d(value: Union[list, tuple, np.ndarray], name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} must be a 3D vector (length 3), got shape {arr.shape}")
    return arr


class MockObstacle:
    """
    Base class for mock obstacles.
    
    All mock obstacles must implement ray_intersect().
    """
    
    def ray_intersect(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        max_distance: float,
    ) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
        """
        Compute ray intersection with this obstacle.
        
        Args:
            origin: Ray origin point, shape=(3,)
            direction: Normalized direction vector, shape=(3,)
            max_distance: Maximum ray distance to consider
            
        Returns:
            (distance, hit_point, normal) if intersection found within max_distance,
            None otherwise.
            - distance: Distance from origin to hit point
            - hit_point: 3D point of intersection, shape=(3,)
            - normal: Surface normal at hit point, shape=(3,)
        """
        raise NotImplementedError("Subclasses must implement ray_intersect()")


class MockSphere(MockObstacle):
    """
    Mock sphere obstacle for testing ray casting.
    
    Implements ray-sphere intersection using standard geometric formula.
    """
    
    def __init__(self, center: Union[list, tuple, np.ndarray], radius: float):
        """
        Args:
            center: Center point of sphere [x, y, z] or shape=(3,) array
            radius: Sphere radius [meters]
        """
        self.center = _to_array3d(center, "center")
        self.radius = float(radius)
    
    def ray_intersect(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        max_distance: float,
    ) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
        """
        Ray-sphere intersection test.
        
        Algorithm: Solve quadratic equation for ray-sphere intersection.
        """
        origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        
        # Vector from sphere center to ray origin
        oc = origin - self.center
        
        # Quadratic coefficients: a*t^2 + b*t + c = 0
        a = np.dot(direction, direction)  # Should be 1.0 if normalized
        b = 2.0 * np.dot(oc, direction)
        c = np.dot(oc, oc) - self.radius * self.radius
        
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return None  # No intersection
        
        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2 * a)
        t2 = (-b + sqrt_discriminant) / (2 * a)
        
        # Find the closest valid intersection
        t = None
        if t1 > 0 and t1 <= max_distance:
            t = t1
        elif t2 > 0 and t2 <= max_distance:
            t = t2
        
        if t is None:
            return None
        
        hit_point = origin + t * direction
        normal = (hit_point - self.center) / self.radius  # Normalized
        
        return (t, hit_point, normal)


class MockPlane(MockObstacle):
    """
    Mock infinite plane obstacle for testing ray casting.
    
    Useful for testing walls and floors.
    """
    
    def __init__(self, point: Union[list, tuple, np.ndarray], normal: Union[list, tuple, np.ndarray]):
        """
        Args:
            point: A point on the plane [x, y, z] or shape=(3,) array
            normal: Plane normal vector [x, y, z] or shape=(3,) array (will be normalized)
        """
        self.point = _to_array3d(point, "point")
        self.normal = _to_array3d(normal, "normal")
        self.normal = self.normal / np.linalg.norm(self.normal)  # Normalize
    
    def ray_intersect(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        max_distance: float,
    ) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
        """
        Ray-plane intersection test.
        
        Algorithm: Solve plane equation for ray parameter t.
        """
        origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        
        # Check if ray is parallel to plane
        denom = np.dot(self.normal, direction)
        if abs(denom) < 1e-10:
            return None  # Ray is parallel to plane
        
        # Compute intersection parameter
        t = np.dot(self.normal, self.point - origin) / denom
        
        if t < 0 or t > max_distance:
            return None  # Intersection is behind origin or too far
        
        hit_point = origin + t * direction
        
        # Normal points in direction of ray (or opposite, depending on convention)
        # Use plane normal directly
        return (t, hit_point, self.normal.copy())


class MockBox(MockObstacle):
    """
    Mock axis-aligned bounding box (AABB) obstacle for testing ray casting.
    
    Useful for testing rectangular obstacles aligned with coordinate axes.
    """
    
    def __init__(self, min_corner: Union[list, tuple, np.ndarray], max_corner: Union[list, tuple, np.ndarray]):
        """
        Args:
            min_corner: Minimum corner [x_min, y_min, z_min] or shape=(3,) array
            max_corner: Maximum corner [x_max, y_max, z_max] or shape=(3,) array
        """
        self.min_corner = _to_array3d(min_corner, "min_corner")
        self.max_corner = _to_array3d(max_corner, "max_corner")
    
    def ray_intersect(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        max_distance: float,
    ) -> Optional[Tuple[float, np.ndarray, np.ndarray]]:
        """
        Ray-AABB intersection test using slab method.
        
        Algorithm: Compute intersection intervals for each axis-aligned slab.
        """
        origin = np.asarray(origin, dtype=np.float64)
        direction = np.asarray(direction, dtype=np.float64)
        
        # Compute intersection intervals for each axis using slab method
        t_min = np.zeros(3, dtype=np.float64)
        t_max = np.zeros(3, dtype=np.float64)
        
        for i in range(3):
            if abs(direction[i]) < 1e-10:
                # Ray is parallel to this axis - check if origin is within bounds
                if origin[i] < self.min_corner[i] or origin[i] > self.max_corner[i]:
                    return None  # Ray is outside the box on this axis
                t_min[i] = -np.inf
                t_max[i] = np.inf
            else:
                inv_dir = 1.0 / direction[i]
                t1 = (self.min_corner[i] - origin[i]) * inv_dir
                t2 = (self.max_corner[i] - origin[i]) * inv_dir
                # Ensure t_min < t_max
                if t1 < t2:
                    t_min[i] = t1
                    t_max[i] = t2
                else:
                    t_min[i] = t2
                    t_max[i] = t1
        
        # Find the largest t_min and smallest t_max
        t_enter = np.max(t_min)
        t_exit = np.min(t_max)
        
        # Check if ray intersects box
        if t_enter > t_exit or t_exit < 0:
            return None  # No intersection
        
        # Use the first intersection point (t_enter)
        t = t_enter if t_enter > 0 else t_exit
        
        if t < 0 or t > max_distance:
            return None  # Intersection is behind origin or too far
        
        hit_point = origin + t * direction
        
        # Compute normal by finding which face was hit
        # Check which axis has the largest component in the intersection
        epsilon = 1e-6
        normal = np.zeros(3, dtype=np.float64)
        
        # Determine which face was hit by checking proximity to box faces
        for i in range(3):
            if abs(hit_point[i] - self.min_corner[i]) < epsilon:
                normal[i] = -1.0
                break
            elif abs(hit_point[i] - self.max_corner[i]) < epsilon:
                normal[i] = 1.0
                break
        
        # Fallback: compute normal from intersection parameter
        if np.allclose(normal, 0):
            # Find which axis contributed most to t_enter
            axis = np.argmax(np.abs(t_min - t_enter))
            # Normal points inward: if entering from min face (direction > 0), normal is -1
            # If entering from max face (direction < 0), normal is +1
            normal[axis] = -1.0 if direction[axis] > 0 else 1.0
        
        return (t, hit_point, normal)


def create_test_obstacles() -> list[MockObstacle]:
    """
    Create a standard set of test obstacles for perception team development.
    
    Returns:
        List of mock obstacles in a simple test scene.
    """
    obstacles = [
        # A sphere at the origin
        MockSphere(center=[0.0, 0.0, 2.0], radius=1.0),
        
        # A floor plane
        MockPlane(
            point=[0.0, 0.0, 0.0],
            normal=[0.0, 0.0, 1.0]  # Upward normal
        ),
        
        # A wall plane (y = 5)
        MockPlane(
            point=[0.0, 5.0, 0.0],
            normal=[0.0, -1.0, 0.0]  # Normal pointing inward
        ),
        
        # A box obstacle
        MockBox(
            min_corner=[-2.0, -2.0, 0.0],
            max_corner=[-1.0, -1.0, 3.0]
        ),
        
        # Another sphere offset
        MockSphere(center=[3.0, 3.0, 1.5], radius=0.5),
    ]
    
    return obstacles

