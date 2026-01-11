"""
Basic tests for Oriented Bounding Box (OBB) collision detection.

Tests verify fundamental OBB collision detection for:
- Axis-aligned boxes (baseline)
- Simple rotations (yaw, pitch, roll)
- Basic collision vs non-collision scenarios
"""

import numpy as np
import pytest
from flockrl_sim import CollisionSystem, Environment, SwarmState
from flockrl_sim.environment.obstacles_types import Wall, RectangularPrism


@pytest.fixture
def basic_environment():
    """Create a basic environment for collision testing."""
    env = Environment(
        bounds=(-20.0, 20.0, -20.0, 20.0, -20.0, 20.0),
        obstacles=[],
        start_position=(-10.0, -10.0, 0.0),
        goal_position=(10.0, 10.0, 0.0),
        seed=42,
    )
    return env


class TestAxisAlignedCollision:
    """Test axis-aligned box collisions (baseline functionality)."""

    def test_wall_collision_from_side(self, basic_environment):
        """Drone approaching axis-aligned wall from the side."""
        wall = Wall(
            id="wall1",
            type="wall",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0),
            length=4.0,
            height=4.0,
            thickness=1.0,
            gate_ids=(),
        )
        basic_environment.obstacles.append(wall)

        collision_system = CollisionSystem(
            environment=basic_environment, drone_radius=0.5
        )

        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[0.0, -1.0, 0.0]]),
            vel=np.array([[0.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0,
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        assert len(collisions) == 1
        assert collisions[0].collision_type == "wall"

    def test_no_collision_outside_range(self, basic_environment):
        """Drone just outside collision range."""
        wall = Wall(
            id="wall1",
            type="wall",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0),
            length=2.0,
            height=2.0,
            thickness=1.0,
            gate_ids=(),
        )
        basic_environment.obstacles.append(wall)

        collision_system = CollisionSystem(
            environment=basic_environment, drone_radius=0.5
        )

        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[0.0, -1.5, 0.0]]),
            vel=np.array([[0.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0,
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        assert len(collisions) == 0

    def test_corner_collision(self, basic_environment):
        """Drone colliding with corner of axis-aligned box."""
        clutter = RectangularPrism(
            id="box1",
            type="RectangularPrism",
            subtype="rectangular_prism",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0),
            length=2.0,
            width=2.0,
            height=2.0,
        )
        basic_environment.obstacles.append(clutter)

        collision_system = CollisionSystem(
            environment=basic_environment, drone_radius=0.5
        )

        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[1.2, 1.2, 1.2]]),
            vel=np.array([[-1.0, -1.0, -1.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0,
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        assert len(collisions) == 1
        assert collisions[0].collision_type == "clutter"


class TestSimpleRotations:
    """Test collision detection with single-axis rotations."""

    def test_45_degree_yaw_rotation(self, basic_environment):
        """Wall rotated 45 degrees around Z axis."""
        yaw = np.pi / 4

        wall = Wall(
            id="wall1",
            type="wall",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, yaw),
            length=4.0,
            height=4.0,
            thickness=0.5,
            gate_ids=(),
        )
        basic_environment.obstacles.append(wall)

        collision_system = CollisionSystem(
            environment=basic_environment, drone_radius=0.5
        )

        # Drone along rotated -Y axis (world diagonal)
        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[0.5, -0.5, 0.0]]),
            vel=np.array([[-1.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0,
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        assert len(collisions) == 1
        assert collisions[0].collision_type == "wall"

    def test_90_degree_yaw_rotation(self, basic_environment):
        """Wall rotated 90 degrees around Z axis."""
        yaw = np.pi / 2

        wall = Wall(
            id="wall1",
            type="wall",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, yaw),
            length=6.0,
            height=4.0,
            thickness=0.5,
            gate_ids=(),
        )
        basic_environment.obstacles.append(wall)

        collision_system = CollisionSystem(
            environment=basic_environment, drone_radius=0.5
        )

        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[0.5, 0.0, 0.0]]),
            vel=np.array([[-1.0, 0.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0,
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        assert len(collisions) == 1

    def test_pitch_rotation(self, basic_environment):
        """Box rotated around Y axis (pitch)."""
        pitch = np.pi / 6

        clutter = RectangularPrism(
            id="box1",
            type="RectangularPrism",
            subtype="rectangular_prism",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, pitch, 0.0),
            length=3.0,
            width=3.0,
            height=3.0,
        )
        basic_environment.obstacles.append(clutter)

        collision_system = CollisionSystem(
            environment=basic_environment, drone_radius=0.5
        )

        # Drone at center - definitely collides
        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[0.0, 0.0, 0.0]]),
            vel=np.array([[-1.0, 0.0, -1.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0,
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        assert len(collisions) == 1

    def test_roll_rotation(self, basic_environment):
        """Box rotated around X axis (roll)."""
        roll = np.pi / 4

        clutter = RectangularPrism(
            id="box1",
            type="RectangularPrism",
            subtype="rectangular_prism",
            position=(0.0, 0.0, 0.0),
            orientation=(roll, 0.0, 0.0),
            length=2.0,
            width=2.0,
            height=2.0,
        )
        basic_environment.obstacles.append(clutter)

        collision_system = CollisionSystem(
            environment=basic_environment, drone_radius=0.5
        )

        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[0.0, 0.0, 0.0]]),
            vel=np.array([[0.0, -1.0, -1.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0,
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        assert len(collisions) == 1


class TestDroneInsideBox:
    """Test collision detection when drone is inside the box."""

    def test_drone_inside_axis_aligned_box(self, basic_environment):
        """Drone center inside axis-aligned box."""
        clutter = RectangularPrism(
            id="box1",
            type="RectangularPrism",
            subtype="rectangular_prism",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0),
            length=4.0,
            width=4.0,
            height=4.0,
        )
        basic_environment.obstacles.append(clutter)

        collision_system = CollisionSystem(
            environment=basic_environment, drone_radius=0.5
        )

        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[0.5, 0.5, 0.5]]),
            vel=np.array([[1.0, 0.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0,
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        assert len(collisions) == 1
        assert collisions[0].penetration_depth > 0

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
            height=4.0,
        )
        basic_environment.obstacles.append(clutter)

        collision_system = CollisionSystem(
            environment=basic_environment, drone_radius=0.5
        )

        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[0.0, 0.0, 0.0]]),
            vel=np.array([[1.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0,
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        assert len(collisions) == 1
        assert collisions[0].penetration_depth > 1.0
