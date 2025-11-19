"""
Tests for OBB collision normal vector computation.

Verifies that collision normals are correctly computed and transformed
for both axis-aligned and rotated boxes.
"""

import numpy as np
import pytest
from flockrl_sim.collision.system import CollisionSystem
from flockrl_sim.environment.obstacles_types import Wall
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
        seed=42,
    )
    return env


class TestAxisAlignedNormals:
    """Test normals for axis-aligned boxes."""

    def test_cardinal_direction_normals(self, basic_environment):
        """Test normals from each cardinal direction."""
        wall = Wall(
            id="wall1",
            type="wall",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0),
            length=2.0,
            height=2.0,
            thickness=2.0,
            gate_ids=(),
        )
        basic_environment.obstacles.append(wall)

        collision_system = CollisionSystem(
            environment=basic_environment, drone_radius=0.5
        )

        # Test each face
        test_cases = [
            (np.array([1.4, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),  # +X face
            (np.array([-1.4, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])),  # -X face
            (np.array([0.0, 1.4, 0.0]), np.array([0.0, 1.0, 0.0])),  # +Y face
            (np.array([0.0, -1.4, 0.0]), np.array([0.0, -1.0, 0.0])),  # -Y face
            (np.array([0.0, 0.0, 1.4]), np.array([0.0, 0.0, 1.0])),  # +Z face
            (np.array([0.0, 0.0, -1.4]), np.array([0.0, 0.0, -1.0])),  # -Z face
        ]

        for pos, expected_normal in test_cases:
            state = SwarmState(
                goals=np.zeros((1, 3)),
                pos=np.array([pos]),
                vel=np.array([[0.0, 0.0, 0.0]]),
                acc=np.array([[0.0, 0.0, 0.0]]),
                ids=np.array([0]),
                t=0.0,
            )

            _, info = collision_system(state)
            collisions = info["collisions"]

            assert len(collisions) == 1, f"No collision at position {pos}"
            normal = collisions[0].normal_vector
            assert np.allclose(normal, expected_normal, atol=1e-6), (
                f"Expected normal {expected_normal}, got {normal} for position {pos}"
            )


class TestRotatedNormals:
    """Test that normals are correctly transformed for rotated boxes."""

    def test_90_degree_yaw_rotation(self, basic_environment):
        """Test normals for box rotated 90 degrees around Z axis."""
        yaw = np.pi / 2

        wall = Wall(
            id="wall1",
            type="wall",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, yaw),
            length=4.0,
            height=4.0,
            thickness=1.0,
            gate_ids=(),
        )
        basic_environment.obstacles.append(wall)

        collision_system = CollisionSystem(
            environment=basic_environment, drone_radius=0.5
        )

        # Drone approaching from +X (which is local Y after rotation)
        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[1.0, 0.0, 0.0]]),
            vel=np.array([[-1.0, 0.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0,
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        assert len(collisions) == 1
        normal = collisions[0].normal_vector

        # Normal should point in +X direction
        assert normal[0] > 0.9, f"Expected normal in +X direction, got {normal}"
