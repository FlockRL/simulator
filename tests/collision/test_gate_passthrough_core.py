"""
Core tests for gate pass-through collision logic.

Tests verify that drones can pass through gates embedded in walls
without triggering wall collisions.
"""

import numpy as np
import pytest
from flockrl_sim.collision.system import CollisionSystem
from flockrl_sim.environment.obstacles_types import Wall, Gate
from flockrl_sim.environment import Environment
from flockrl_sim.state import SwarmState


@pytest.fixture
def simple_environment():
    """Create a simple environment with one wall containing a gate."""
    wall = Wall(
        id="wall1",
        type="wall",
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        length=10.0,
        height=8.0,
        thickness=0.2,
        gate_ids=("gate1",),
    )

    gate = Gate(
        id="gate1",
        type="gate",
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        width=2.0,
        height=2.0,
        thickness=0.2,
    )

    env = Environment(
        bounds=(-10.0, 10.0, -10.0, 10.0, -10.0, 10.0),
        obstacles=[gate, wall],
        start_position=(-5.0, -5.0, 0.0),
        goal_position=(5.0, 5.0, 0.0),
        seed=42,
    )

    return env


@pytest.fixture
def collision_system(simple_environment):
    """Create a collision system with the simple environment."""
    return CollisionSystem(environment=simple_environment, drone_radius=0.5)


class TestBasicGatePassThrough:
    """Test basic gate pass-through functionality."""

    def test_drone_inside_gate_no_collision(self, collision_system):
        """Drone centered in gate should not collide with wall."""
        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[0.0, 0.0, 0.0]]),
            vel=np.array([[1.0, 0.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0,
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        wall_collisions = [c for c in collisions if c.collision_type == "wall"]
        assert len(wall_collisions) == 0

    def test_drone_outside_gate_does_collide(self, collision_system):
        """Drone outside gate volume should collide with wall."""
        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[2.0, 0.0, 0.0]]),
            vel=np.array([[0.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0,
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        wall_collisions = [c for c in collisions if c.collision_type == "wall"]
        assert len(wall_collisions) == 1
        assert wall_collisions[0].drone_id == 0


class TestGateBoundaries:
    """Test gate boundary conditions."""

    def test_drone_near_gate_edge_no_collision(self, collision_system):
        """Drone near gate edge but still inside should not collide."""
        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[0.9, 0.0, 0.9]]),
            vel=np.array([[0.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0,
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        wall_collisions = [c for c in collisions if c.collision_type == "wall"]
        assert len(wall_collisions) == 0

    def test_drone_above_gate_does_collide(self, collision_system):
        """Drone above gate height should collide with wall."""
        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[0.0, 0.0, 2.0]]),
            vel=np.array([[0.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0,
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        wall_collisions = [c for c in collisions if c.collision_type == "wall"]
        assert len(wall_collisions) == 1

    def test_drone_below_gate_does_collide(self, collision_system):
        """Drone below gate height should collide with wall."""
        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[0.0, 0.0, -2.0]]),
            vel=np.array([[0.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0,
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        wall_collisions = [c for c in collisions if c.collision_type == "wall"]
        assert len(wall_collisions) == 1


class TestMultipleDrones:
    """Test gate pass-through with multiple drones."""

    def test_multiple_drones_mixed_positions(self, collision_system):
        """Test multiple drones, some inside gate, some outside."""
        state = SwarmState(
            goals=np.zeros((4, 3)),
            pos=np.array(
                [
                    [0.0, 0.0, 0.0],  # Inside gate
                    [3.0, 0.0, 0.0],  # Outside gate, colliding
                    [0.5, 0.0, 0.5],  # Inside gate
                    [-3.0, 0.0, 0.0],  # Outside gate, colliding
                ]
            ),
            vel=np.array(
                [
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ]
            ),
            acc=np.zeros((4, 3)),
            ids=np.array([0, 1, 2, 3]),
            t=0.0,
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        wall_collisions = [c for c in collisions if c.collision_type == "wall"]
        colliding_ids = {c.drone_id for c in wall_collisions}

        assert colliding_ids == {1, 3}


class TestWallWithoutGate:
    """Test that walls without gates work normally."""

    def test_wall_without_gate_collides(self):
        """Wall without gates should collide normally."""
        wall = Wall(
            id="wall1",
            type="wall",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0),
            length=10.0,
            height=8.0,
            thickness=0.2,
            gate_ids=(),
        )

        env = Environment(
            bounds=(-10.0, 10.0, -10.0, 10.0, -10.0, 10.0),
            obstacles=[wall],
            start_position=(-5.0, -5.0, 0.0),
            goal_position=(5.0, 5.0, 0.0),
            seed=42,
        )

        collision_system = CollisionSystem(environment=env, drone_radius=0.5)

        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[0.0, 0.0, 0.0]]),
            vel=np.array([[0.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0,
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        wall_collisions = [c for c in collisions if c.collision_type == "wall"]
        assert len(wall_collisions) == 1
