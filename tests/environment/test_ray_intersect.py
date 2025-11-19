"""
Tests for ray intersection functionality with obstacles.

This module tests the ray_intersect methods for various obstacle types
including RectangularPrism, Wall, and Gate objects.
"""

import numpy as np
import pytest
from math import isclose

from flockrl_sim.environment.obstacles_types import Gate, Wall, RectangularPrism


def unit(v):
    """Normalize a vector to unit length."""
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v)


def assert_vec_close(a, b, tol=1e-6):
    """Assert that two vectors are close within a tolerance."""
    a = np.array(a, float)
    b = np.array(b, float)
    assert np.allclose(a, b, atol=tol), f"Vectors differ: {a} vs {b}"


class TestRectangularPrismRayIntersect:
    """Tests for RectangularPrism ray intersection."""

    def test_axis_aligned_hits(self):
        """Test ray intersections with an axis-aligned box."""
        box = RectangularPrism(
            id="box1",
            type="box",
            position=(0, 0, 0),
            orientation=(0, 0, 0),
            length=2,
            width=2,
            height=2,
            subtype="box",
        )

        # Test hitting top face (Z+)
        d = unit((0, 0, -1))
        hit = box.ray_intersect(np.array([0, 0, 5], float), d, 100)
        assert hit is not None
        dist, p, n = hit
        assert isclose(dist, 4)
        assert_vec_close(p, [0, 0, 1])
        assert_vec_close(n, [0, 0, 1])

        # Test hitting right face (X+)
        d = unit((-1, 0, 0))
        hit = box.ray_intersect(np.array([5, 0, 0], float), d, 100)
        assert hit is not None
        dist, p, n = hit
        assert isclose(dist, 4)
        assert_vec_close(p, [1, 0, 0])
        assert_vec_close(n, [1, 0, 0])

        # Test hitting front face (Y+)
        d = unit((0, -1, 0))
        hit = box.ray_intersect(np.array([0, 5, 0], float), d, 100)
        assert hit is not None
        dist, p, n = hit
        assert isclose(dist, 4)
        assert_vec_close(p, [0, 1, 0])
        assert_vec_close(n, [0, 1, 0])

    def test_miss(self):
        """Test rays that miss the box."""
        box = RectangularPrism(
            id="box2",
            type="box",
            position=(0, 0, 0),
            orientation=(0, 0, 0),
            length=2,
            width=2,
            height=2,
            subtype="box",
        )

        # Ray parallel to box, outside its bounds
        d = unit((1, 0, 0))
        hit = box.ray_intersect(np.array([-5, 5, 0], float), d, 100)
        assert hit is None

        # Ray perpendicular, missing the box
        d = unit((0, 1, 0))
        hit = box.ray_intersect(np.array([3, 0, 0], float), d, 100)
        assert hit is None

    def test_rotated(self):
        """Test ray intersection with a rotated box."""
        box = RectangularPrism(
            id="box3",
            type="box",
            position=(0, 0, 0),
            orientation=(0, 0, np.pi / 2),
            length=2,
            width=2,
            height=2,
            subtype="box",
        )

        d = unit((-1, 0, 0))
        hit = box.ray_intersect(np.array([5, 0, 0], float), d, 100)
        assert hit is not None

        dist, p, n = hit
        assert isclose(dist, 4)
        # Normal should be in world coordinates, not local
        assert_vec_close(n, [1, 0, 0])

    def test_grazing(self):
        """Test a ray that grazes the edge of a box."""
        box = RectangularPrism(
            id="box4",
            type="box",
            position=(0, 0, 0),
            orientation=(0, 0, 0),
            length=2,
            width=2,
            height=2,
            subtype="box",
        )

        # Ray just inside the +X face boundary
        origin = np.array([1 - 1e-7, 0, 5], float)
        d = unit((0, 0, -1))

        hit = box.ray_intersect(origin, d, 100)
        assert hit is not None


class TestWallRayIntersect:
    """Tests for Wall ray intersection."""

    def test_basic(self):
        """Test basic ray intersections with an axis-aligned wall."""
        wall = Wall(
            id="w1",
            type="wall",
            position=(0, 0, 0),
            orientation=(0, 0, 0),
            length=10,
            thickness=0.5,
            height=3,
        )

        # Ray hitting the wall from the side
        d = unit((-1, 0, 0))
        hit = wall.ray_intersect(np.array([20, 0, 0], float), d, 100)
        assert hit is not None
        dist, p, n = hit
        # Wall extends from -5 to +5 in X, thickness is 0.5, so edge is at 5 + 0.25 = 5.25
        # Actually: length/2 = 5, thickness/2 = 0.25, so X extent is [-5, 5]
        # Ray from (20, 0, 0) going left hits at X=5, but wall has thickness
        # The half-sizes are [5, 0.25, 1.5], so face is at 5 in local X
        # Distance should be 20 - 5 = 15
        assert isclose(dist, 15)
        assert_vec_close(p, [5, 0, 0])
        assert_vec_close(n, [1, 0, 0])

        # Ray hitting the wall from above
        d = unit((0, 0, -1))
        hit = wall.ray_intersect(np.array([0, 0, 10], float), d, 100)
        assert hit is not None
        dist, p, n = hit
        # Wall height is 3, so Z extent is [-1.5, 1.5]
        # Ray from (0, 0, 10) hits at Z=1.5
        # Distance should be 10 - 1.5 = 8.5
        assert isclose(dist, 8.5)
        assert_vec_close(p, [0, 0, 1.5])
        assert_vec_close(n, [0, 0, 1])

    def test_rotated(self):
        """Test ray intersection with a rotated wall."""
        wall = Wall(
            id="w2",
            type="wall",
            position=(0, 0, 0),
            orientation=(0, 0, np.pi / 2),
            length=10,
            thickness=0.5,
            height=3,
        )

        # After 90° rotation around Z, the wall's length is along Y
        d = unit((0, -1, 0))
        hit = wall.ray_intersect(np.array([0, 20, 0], float), d, 100)
        assert hit is not None
        dist, p, n = hit
        assert isclose(dist, 15)
        assert_vec_close(p, [0, 5, 0])
        assert_vec_close(n, [0, 1, 0])


class TestGateRayIntersect:
    """Tests for Gate ray intersection."""

    def test_basic(self):
        """Test basic ray intersections with an axis-aligned gate."""
        gate = Gate(
            id="g1",
            type="gate",
            position=(0, 0, 0),
            orientation=(0, 0, 0),
            width=4,
            height=5,
            thickness=0.5,
        )

        # Ray hitting the gate from the side
        d = unit((-1, 0, 0))
        hit = gate.ray_intersect(np.array([10, 0, 0], float), d, 100)
        assert hit is not None
        dist, p, n = hit
        # Gate width is 4, so X extent is [-2, 2]
        # Distance from (10, 0, 0) to X=2 is 8
        assert isclose(dist, 8)
        assert_vec_close(p, [2, 0, 0])
        assert_vec_close(n, [1, 0, 0])

        # Ray hitting the gate from above
        d = unit((0, 0, -1))
        hit = gate.ray_intersect(np.array([0, 0, 10], float), d, 100)
        assert hit is not None
        dist, p, n = hit
        # Gate height is 5, so Z extent is [-2.5, 2.5]
        # Distance from (0, 0, 10) to Z=2.5 is 7.5
        assert isclose(dist, 7.5)
        assert_vec_close(p, [0, 0, 2.5])
        assert_vec_close(n, [0, 0, 1])

    def test_rotated(self):
        """Test ray intersection with a rotated gate."""
        gate = Gate(
            id="g2",
            type="gate",
            position=(0, 0, 0),
            orientation=(0, np.pi / 2, 0),
            width=4,
            height=5,
            thickness=0.5,
        )

        # After 90° rotation around Y, the gate's width is along Z
        d = unit((1, 0, 0))
        hit = gate.ray_intersect(np.array([-10, 0, 0], float), d, 100)
        assert hit is not None
        dist, p, n = hit
        # After rotation, the thickness (Y) extent determines the X hit position
        # Distance should be 10 - 2.5 = 7.5 (half of height which is now along X axis due to rotation)
        assert isclose(dist, 7.5)
        assert_vec_close(p, [-2.5, 0, 0])
