"""
Tests for Oriented Bounding Box (OBB) collision detection.

Tests verify that OBB collision detection works correctly for:
- Axis-aligned boxes (special case)
- Rotated boxes in various orientations
- Drones approaching from different angles
- Edge cases (drone inside box, grazing collisions, etc.)
"""

import numpy as np
import pytest
from flockrl_sim.collision.system import CollisionSystem
from flockrl_sim.environment.obstacles_types import Wall, RectangularPrism
from flockrl_sim.environment import Environment
from flockrl_sim.state import SwarmState


@pytest.fixture
def basic_environment():
    """Create a basic environment for collision testing."""
    env = Environment(
        bounds=(-20.0, 20.0, -20.0, 20.0, -20.0, 20.0),
        obstacles=[],
        start_position=(-10.0, -10.0, 0.0),
        goal_position=(10.0, 10.0, 0.0),
        seed=42
    )
    return env


class TestAxisAlignedBoxCollision:
    """Test that axis-aligned boxes still work correctly (regression tests)."""

    def test_aabb_collision_from_side(self, basic_environment):
        """Drone approaching axis-aligned box from the side."""
        wall = Wall(
            id="wall1",
            type="wall",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0),
            length=4.0,
            height=4.0,
            thickness=1.0,
            gate_ids=()
        )
        basic_environment.obstacles.append(wall)

        collision_system = CollisionSystem(environment=basic_environment, drone_radius=0.5)

        # Drone approaching from -Y direction
        state = SwarmState(
            pos=np.array([[0.0, -1.0, 0.0]]),  # Within collision range
            vel=np.array([[0.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        assert len(collisions) == 1
        assert collisions[0].collision_type == "wall"
        assert collisions[0].drone_id == 0

    def test_aabb_no_collision(self, basic_environment):
        """Drone near but not colliding with axis-aligned box."""
        wall = Wall(
            id="wall1",
            type="wall",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0),
            length=2.0,
            height=2.0,
            thickness=1.0,
            gate_ids=()
        )
        basic_environment.obstacles.append(wall)

        collision_system = CollisionSystem(environment=basic_environment, drone_radius=0.5)

        # Drone just outside collision range
        state = SwarmState(
            pos=np.array([[0.0, -1.5, 0.0]]),  # 1.5m from center, box extends 0.5m, drone radius 0.5m
            vel=np.array([[0.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        assert len(collisions) == 0

    def test_aabb_corner_collision(self, basic_environment):
        """Drone colliding with corner of axis-aligned box."""
        clutter = RectangularPrism(
            id="box1",
            type="RectangularPrism",  # Fixed: was "clutter", should be "RectangularPrism"
            subtype="rectangular_prism",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0),
            length=2.0,
            width=2.0,
            height=2.0
        )
        basic_environment.obstacles.append(clutter)

        collision_system = CollisionSystem(environment=basic_environment, drone_radius=0.5)

        # Drone near corner (distance from corner = sqrt(0.2^2 * 3) = 0.346 < 0.5)
        state = SwarmState(
            pos=np.array([[1.2, 1.2, 1.2]]),  # Near (+,+,+) corner
            vel=np.array([[-1.0, -1.0, -1.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        assert len(collisions) == 1
        assert collisions[0].collision_type == "clutter"


class TestRotatedBoxCollision:
    """Test collision detection with rotated boxes."""

    def test_wall_rotated_45deg_yaw(self, basic_environment):
        """Wall rotated 45 degrees around Z axis (yaw)."""
        yaw = np.pi / 4  # 45 degrees

        wall = Wall(
            id="wall1",
            type="wall",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, yaw),
            length=4.0,
            height=4.0,
            thickness=0.5,
            gate_ids=()
        )
        basic_environment.obstacles.append(wall)

        collision_system = CollisionSystem(environment=basic_environment, drone_radius=0.5)

        # Drone approaching rotated wall from local -Y direction (world diagonal)
        # Local Y-axis points at 45 degrees in world space
        # So position along world diagonal should collide
        state = SwarmState(
            pos=np.array([[0.5, -0.5, 0.0]]),  # Along rotated -Y axis
            vel=np.array([[-1.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        assert len(collisions) == 1
        assert collisions[0].collision_type == "wall"
        # Normal should be rotated too
        assert len(collisions[0].normal_vector) == 3

    def test_wall_rotated_90deg_yaw(self, basic_environment):
        """Wall rotated 90 degrees around Z axis."""
        yaw = np.pi / 2  # 90 degrees

        wall = Wall(
            id="wall1",
            type="wall",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, yaw),
            length=6.0,
            height=4.0,
            thickness=0.5,
            gate_ids=()
        )
        basic_environment.obstacles.append(wall)

        collision_system = CollisionSystem(environment=basic_environment, drone_radius=0.5)

        # After 90 degree rotation, local X becomes world -Y
        # So length extends along Y axis
        state = SwarmState(
            pos=np.array([[0.5, 0.0, 0.0]]),  # Should collide
            vel=np.array([[-1.0, 0.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        assert len(collisions) == 1

    def test_box_rotated_pitch(self, basic_environment):
        """Box rotated around Y axis (pitch)."""
        pitch = np.pi / 6  # 30 degrees

        clutter = RectangularPrism(
            id="box1",
            type="RectangularPrism",
            subtype="rectangular_prism",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, pitch, 0.0),
            length=3.0,
            width=3.0,
            height=3.0
        )
        basic_environment.obstacles.append(clutter)

        collision_system = CollisionSystem(environment=basic_environment, drone_radius=0.5)

        # Drone at center of box - definitely collides regardless of rotation
        state = SwarmState(
            pos=np.array([[0.0, 0.0, 0.0]]),
            vel=np.array([[-1.0, 0.0, -1.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        # Should detect collision with rotated box
        assert len(collisions) == 1

    def test_box_rotated_roll(self, basic_environment):
        """Box rotated around X axis (roll)."""
        roll = np.pi / 4  # 45 degrees

        clutter = RectangularPrism(
            id="box1",
            type="RectangularPrism",
            subtype="rectangular_prism",
            position=(0.0, 0.0, 0.0),
            orientation=(roll, 0.0, 0.0),
            length=2.0,
            width=2.0,
            height=2.0
        )
        basic_environment.obstacles.append(clutter)

        collision_system = CollisionSystem(environment=basic_environment, drone_radius=0.5)

        # Drone at center - simpler test
        state = SwarmState(
            pos=np.array([[0.0, 0.0, 0.0]]),
            vel=np.array([[0.0, -1.0, -1.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        assert len(collisions) == 1

    def test_box_compound_rotation(self, basic_environment):
        """Box with rotation around multiple axes."""
        roll = np.pi / 6
        pitch = np.pi / 8
        yaw = np.pi / 4

        clutter = RectangularPrism(
            id="box1",
            type="RectangularPrism",
            subtype="rectangular_prism",
            position=(5.0, 5.0, 5.0),
            orientation=(roll, pitch, yaw),
            length=2.0,
            width=2.0,
            height=2.0
        )
        basic_environment.obstacles.append(clutter)

        collision_system = CollisionSystem(environment=basic_environment, drone_radius=0.5)

        # Drone at box center should definitely collide
        state = SwarmState(
            pos=np.array([[5.0, 5.0, 5.0]]),
            vel=np.array([[1.0, 0.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        assert len(collisions) == 1
        assert collisions[0].penetration_depth > 0


class TestCollisionNormals:
    """Test that collision normals are correctly computed and transformed."""

    def test_aabb_normal_directions(self, basic_environment):
        """Test normals for axis-aligned box from each cardinal direction."""
        wall = Wall(
            id="wall1",
            type="wall",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0),
            length=2.0,
            height=2.0,
            thickness=2.0,
            gate_ids=()
        )
        basic_environment.obstacles.append(wall)

        collision_system = CollisionSystem(environment=basic_environment, drone_radius=0.5)

        # Test each face - positions must be within drone_radius of surface
        # Wall dims: length=2.0 (x: ±1.0), thickness=2.0 (y: ±1.0), height=2.0 (z: ±1.0)
        # For collision: distance <= 0.5, so position at ±1.5 should collide
        test_cases = [
            (np.array([1.4, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),    # +X face
            (np.array([-1.4, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])),  # -X face
            (np.array([0.0, 1.4, 0.0]), np.array([0.0, 1.0, 0.0])),    # +Y face
            (np.array([0.0, -1.4, 0.0]), np.array([0.0, -1.0, 0.0])),  # -Y face
            (np.array([0.0, 0.0, 1.4]), np.array([0.0, 0.0, 1.0])),    # +Z face
            (np.array([0.0, 0.0, -1.4]), np.array([0.0, 0.0, -1.0])),  # -Z face
        ]

        for pos, expected_normal in test_cases:
            state = SwarmState(
                pos=np.array([pos]),
                vel=np.array([[0.0, 0.0, 0.0]]),
                acc=np.array([[0.0, 0.0, 0.0]]),
                ids=np.array([0]),
                t=0.0
            )

            _, info = collision_system(state)
            collisions = info["collisions"]

            assert len(collisions) == 1, f"No collision at position {pos}"
            normal = collisions[0].normal_vector
            # Normals should point away from box
            assert np.allclose(normal, expected_normal, atol=1e-6), \
                f"Expected normal {expected_normal}, got {normal} for position {pos}"

    def test_rotated_box_normal_transformed(self, basic_environment):
        """Test that normals are correctly transformed for rotated boxes."""
        yaw = np.pi / 2  # 90 degrees

        wall = Wall(
            id="wall1",
            type="wall",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, yaw),
            length=4.0,
            height=4.0,
            thickness=1.0,
            gate_ids=()
        )
        basic_environment.obstacles.append(wall)

        collision_system = CollisionSystem(environment=basic_environment, drone_radius=0.5)

        # Drone approaching from what would be local -Y direction
        # After 90 degree yaw rotation, local Y becomes world X
        state = SwarmState(
            pos=np.array([[1.0, 0.0, 0.0]]),
            vel=np.array([[-1.0, 0.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        assert len(collisions) == 1
        normal = collisions[0].normal_vector

        # Normal should be rotated too (local -Y becomes world -X after 90 deg yaw)
        # Actually, local Y becomes world -X, so normal in local Y is world -X
        # But the drone is at world +X, so normal should point in +X
        assert normal[0] > 0.9  # Primarily in +X direction


class TestDroneInsideBox:
    """Test collision detection when drone center is inside the box."""

    def test_drone_inside_aabb(self, basic_environment):
        """Drone center inside axis-aligned box."""
        clutter = RectangularPrism(
            id="box1",
            type="RectangularPrism",
            subtype="rectangular_prism",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0),
            length=4.0,
            width=4.0,
            height=4.0
        )
        basic_environment.obstacles.append(clutter)

        collision_system = CollisionSystem(environment=basic_environment, drone_radius=0.5)

        # Drone center well inside box
        state = SwarmState(
            pos=np.array([[0.5, 0.5, 0.5]]),
            vel=np.array([[1.0, 0.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        assert len(collisions) == 1
        assert collisions[0].penetration_depth > 0
        # Should push drone out through nearest face

    def test_drone_inside_rotated_box(self, basic_environment):
        """Drone center inside rotated box."""
        clutter = RectangularPrism(
            id="box1",
            type="RectangularPrism",
            subtype="rectangular_prism",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, np.pi / 4),
            length=4.0,
            width=4.0,
            height=4.0
        )
        basic_environment.obstacles.append(clutter)

        collision_system = CollisionSystem(environment=basic_environment, drone_radius=0.5)

        state = SwarmState(
            pos=np.array([[0.0, 0.0, 0.0]]),  # At center
            vel=np.array([[1.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        assert len(collisions) == 1
        # Large penetration since drone is at center
        assert collisions[0].penetration_depth > 1.0


class TestMultipleRotatedObstacles:
    """Test scenes with multiple rotated obstacles."""

    def test_two_rotated_walls_different_angles(self, basic_environment):
        """Two walls at different rotation angles."""
        wall1 = Wall(
            id="wall1",
            type="wall",
            position=(-3.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0),
            length=2.0,
            height=4.0,
            thickness=0.5,
            gate_ids=()
        )

        wall2 = Wall(
            id="wall2",
            type="wall",
            position=(3.0, 0.0, 0.0),
            orientation=(0.0, 0.0, np.pi / 3),  # 60 degrees
            length=2.0,
            height=4.0,
            thickness=0.5,
            gate_ids=()
        )

        basic_environment.obstacles.extend([wall1, wall2])
        collision_system = CollisionSystem(environment=basic_environment, drone_radius=0.5)

        # Drone colliding with wall1 but not wall2
        state = SwarmState(
            pos=np.array([[-3.0, -0.5, 0.0]]),
            vel=np.array([[0.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        wall_collisions = [c for c in collisions if c.collision_type == "wall"]
        assert len(wall_collisions) == 1

    def test_maze_of_rotated_boxes(self, basic_environment):
        """Multiple rotated boxes at various angles."""
        boxes = [
            RectangularPrism(
                id=f"box{i}",
                type="RectangularPrism",
                subtype="rectangular_prism",
                position=(i * 3.0, 0.0, 0.0),
                orientation=(0.0, 0.0, i * np.pi / 8),
                length=1.5,
                width=1.5,
                height=2.0
            )
            for i in range(4)
        ]

        basic_environment.obstacles.extend(boxes)
        collision_system = CollisionSystem(environment=basic_environment, drone_radius=0.5)

        # Drone passing through without hitting any
        state = SwarmState(
            pos=np.array([[1.5, 2.0, 0.0]]),
            vel=np.array([[1.0, 0.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        assert len(collisions) == 0  # Should miss all boxes


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_grazing_collision_rotated(self, basic_environment):
        """Drone barely touching rotated box."""
        clutter = RectangularPrism(
            id="box1",
            type="RectangularPrism",
            subtype="rectangular_prism",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, np.pi / 4),
            length=2.0,
            width=2.0,
            height=2.0
        )
        basic_environment.obstacles.append(clutter)

        collision_system = CollisionSystem(environment=basic_environment, drone_radius=0.5)

        # Simpler test: drone close to surface
        state = SwarmState(
            pos=np.array([[1.4, 0.0, 0.0]]),  # Close to box surface
            vel=np.array([[-1.0, 0.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        assert len(collisions) == 1
        assert collisions[0].penetration_depth < 0.6  # Allow some numerical tolerance

    def test_null_orientation_same_as_zeros(self, basic_environment):
        """None orientation should behave same as (0, 0, 0)."""
        wall1 = Wall(
            id="wall1",
            type="wall",
            position=(-2.0, 0.0, 0.0),
            orientation=None,
            length=2.0,
            height=2.0,
            thickness=1.0,
            gate_ids=()
        )

        wall2 = Wall(
            id="wall2",
            type="wall",
            position=(2.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0),
            length=2.0,
            height=2.0,
            thickness=1.0,
            gate_ids=()
        )

        basic_environment.obstacles.extend([wall1, wall2])
        collision_system = CollisionSystem(environment=basic_environment, drone_radius=0.5)

        # Test both walls - thickness is 1.0, so extends to y=±0.5
        # Drone at y=-0.9 with radius 0.5 should collide
        for x_pos in [-2.0, 2.0]:
            state = SwarmState(
                pos=np.array([[x_pos, -0.9, 0.0]]),
                vel=np.array([[0.0, 1.0, 0.0]]),
                acc=np.array([[0.0, 0.0, 0.0]]),
                ids=np.array([0]),
                t=0.0
            )

            _, info = collision_system(state)
            collisions = info["collisions"]

            assert len(collisions) == 1

    def test_very_small_rotation_angle(self, basic_environment):
        """Box with very small rotation angle (near zero)."""
        clutter = RectangularPrism(
            id="box1",
            type="RectangularPrism",
            subtype="rectangular_prism",
            position=(0.0, 0.0, 0.0),
            orientation=(1e-10, 1e-10, 1e-10),
            length=2.0,
            width=2.0,
            height=2.0
        )
        basic_environment.obstacles.append(clutter)

        collision_system = CollisionSystem(environment=basic_environment, drone_radius=0.5)

        state = SwarmState(
            pos=np.array([[1.4, 0.0, 0.0]]),  # Close to surface
            vel=np.array([[-1.0, 0.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        # Should work just like axis-aligned
        assert len(collisions) == 1
