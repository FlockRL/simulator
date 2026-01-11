"""
Complex OBB collision scenarios and edge cases.

Tests for compound rotations, multiple obstacles, and boundary conditions.
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


class TestCompoundRotations:
    """Test boxes with multiple rotation axes."""

    def test_triple_axis_rotation(self, basic_environment):
        """Box with rotation around all three axes."""
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
            height=2.0,
        )
        basic_environment.obstacles.append(clutter)

        collision_system = CollisionSystem(
            environment=basic_environment, drone_radius=0.5, restitution=1.0
        )

        # Drone at box center should collide
        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[5.0, 5.0, 5.0]]),
            vel=np.array([[1.0, 0.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0,
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        assert len(collisions) == 1
        assert collisions[0].penetration_depth > 0


class TestMultipleObstacles:
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
            gate_ids=(),
        )

        wall2 = Wall(
            id="wall2",
            type="wall",
            position=(3.0, 0.0, 0.0),
            orientation=(0.0, 0.0, np.pi / 3),
            length=2.0,
            height=4.0,
            thickness=0.5,
            gate_ids=(),
        )

        basic_environment.obstacles.extend([wall1, wall2])
        collision_system = CollisionSystem(
            environment=basic_environment, drone_radius=0.5, restitution=1.0
        )

        # Drone colliding with wall1 but not wall2
        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[-3.0, -0.5, 0.0]]),
            vel=np.array([[0.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0,
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        wall_collisions = [c for c in collisions if c.collision_type == "wall"]
        assert len(wall_collisions) == 1

    def test_multiple_rotated_boxes(self, basic_environment):
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
                height=2.0,
            )
            for i in range(4)
        ]

        basic_environment.obstacles.extend(boxes)
        collision_system = CollisionSystem(
            environment=basic_environment, drone_radius=0.5, restitution=1.0
        )

        # Drone passing through without hitting any
        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[1.5, 2.0, 0.0]]),
            vel=np.array([[1.0, 0.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0,
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        assert len(collisions) == 0


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
            height=2.0,
        )
        basic_environment.obstacles.append(clutter)

        collision_system = CollisionSystem(
            environment=basic_environment, drone_radius=0.5, restitution=1.0
        )

        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[1.4, 0.0, 0.0]]),
            vel=np.array([[-1.0, 0.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0,
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        assert len(collisions) == 1
        assert collisions[0].penetration_depth < 0.6

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
            height=2.0,
        )
        basic_environment.obstacles.append(clutter)

        collision_system = CollisionSystem(
            environment=basic_environment, drone_radius=0.5, restitution=1.0
        )

        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[1.4, 0.0, 0.0]]),
            vel=np.array([[-1.0, 0.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0,
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        assert len(collisions) == 1
