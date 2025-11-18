"""
Edge cases and complex scenarios for gate pass-through logic.

Tests for multiple gates, rotated gates, and error handling.
"""

import numpy as np
import pytest
from flockrl_sim.collision.system import CollisionSystem
from flockrl_sim.environment.obstacles_types import Wall, Gate
from flockrl_sim.environment import Environment
from flockrl_sim.state import SwarmState


class TestMultipleGatesInWall:
    """Test walls with multiple gates."""

    def test_wall_with_two_gates(self):
        """Test a wall with two separate gates."""
        wall = Wall(
            id="wall1",
            type="wall",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0),
            length=10.0,
            height=8.0,
            thickness=0.2,
            gate_ids=("gate1", "gate2")
        )

        gate1 = Gate(
            id="gate1",
            type="gate",
            position=(-2.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0),
            width=1.5,
            height=2.0,
            thickness=0.2
        )

        gate2 = Gate(
            id="gate2",
            type="gate",
            position=(2.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0),
            width=1.5,
            height=2.0,
            thickness=0.2
        )

        env = Environment(
            bounds=(-10.0, 10.0, -10.0, 10.0, -10.0, 10.0),
            obstacles=[gate1, gate2, wall],
            start_position=(-5.0, -5.0, 0.0),
            goal_position=(5.0, 5.0, 0.0),
            seed=42
        )

        collision_system = CollisionSystem(environment=env, drone_radius=0.5)

        state = SwarmState(
            goals=np.zeros((3, 3)),
            pos=np.array([
                [-2.0, 0.0, 0.0],   # Inside gate1
                [2.0, 0.0, 0.0],    # Inside gate2
                [0.0, 0.0, 0.0],    # Between gates (should collide)
            ]),
            vel=np.array([
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]),
            acc=np.zeros((3, 3)),
            ids=np.array([0, 1, 2]),
            t=0.0
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        wall_collisions = [c for c in collisions if c.collision_type == "wall"]
        colliding_ids = {c.drone_id for c in wall_collisions}

        assert colliding_ids == {2}


class TestRotatedGates:
    """Test gates with non-zero orientation (rotated gates)."""

    def test_rotated_gate_passthrough(self):
        """Test that rotated gates correctly allow pass-through."""
        yaw = np.pi / 4

        wall = Wall(
            id="wall1",
            type="wall",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, yaw),
            length=10.0,
            height=8.0,
            thickness=0.2,
            gate_ids=("gate1",)
        )

        gate = Gate(
            id="gate1",
            type="gate",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, yaw),
            width=2.0,
            height=2.0,
            thickness=0.2
        )

        env = Environment(
            bounds=(-10.0, 10.0, -10.0, 10.0, -10.0, 10.0),
            obstacles=[gate, wall],
            start_position=(-5.0, -5.0, 0.0),
            goal_position=(5.0, 5.0, 0.0),
            seed=42
        )

        collision_system = CollisionSystem(environment=env, drone_radius=0.5)

        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[0.0, 0.0, 0.0]]),
            vel=np.array([[1.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        wall_collisions = [c for c in collisions if c.collision_type == "wall"]
        assert len(wall_collisions) == 0


class TestDroneRadiusConsideration:
    """Test that drone radius is properly considered."""

    def test_drone_radius_with_small_gate(self):
        """Verify drone radius consideration in gate pass-through."""
        wall = Wall(
            id="wall1",
            type="wall",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0),
            length=10.0,
            height=8.0,
            thickness=0.2,
            gate_ids=("gate1",)
        )

        gate = Gate(
            id="gate1",
            type="gate",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0),
            width=1.0,
            height=1.0,
            thickness=0.2
        )

        env = Environment(
            bounds=(-10.0, 10.0, -10.0, 10.0, -10.0, 10.0),
            obstacles=[gate, wall],
            start_position=(-5.0, -5.0, 0.0),
            goal_position=(5.0, 5.0, 0.0),
            seed=42
        )

        collision_system = CollisionSystem(environment=env, drone_radius=0.6)

        # Drone center at gate edge
        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[0.5, 0.0, 0.0]]),
            vel=np.array([[0.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        wall_collisions = [c for c in collisions if c.collision_type == "wall"]
        assert len(wall_collisions) == 0


class TestErrorHandling:
    """Test edge cases and error handling."""

    def test_empty_gate_ids(self):
        """Wall with empty gate_ids list should behave like no gates."""
        wall = Wall(
            id="wall1",
            type="wall",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0),
            length=10.0,
            height=8.0,
            thickness=0.2,
            gate_ids=()
        )

        env = Environment(
            bounds=(-10.0, 10.0, -10.0, 10.0, -10.0, 10.0),
            obstacles=[wall],
            start_position=(-5.0, -5.0, 0.0),
            goal_position=(5.0, 5.0, 0.0),
            seed=42
        )

        collision_system = CollisionSystem(environment=env, drone_radius=0.5)

        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[0.0, 0.0, 0.0]]),
            vel=np.array([[0.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        wall_collisions = [c for c in collisions if c.collision_type == "wall"]
        assert len(wall_collisions) == 1

    def test_missing_gate_in_map(self):
        """Wall references gate ID that doesn't exist in obstacle list."""
        wall = Wall(
            id="wall1",
            type="wall",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0),
            length=10.0,
            height=8.0,
            thickness=0.2,
            gate_ids=("nonexistent_gate",)
        )

        env = Environment(
            bounds=(-10.0, 10.0, -10.0, 10.0, -10.0, 10.0),
            obstacles=[wall],
            start_position=(-5.0, -5.0, 0.0),
            goal_position=(5.0, 5.0, 0.0),
            seed=42
        )

        collision_system = CollisionSystem(environment=env, drone_radius=0.5)

        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[0.0, 0.0, 0.0]]),
            vel=np.array([[0.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        wall_collisions = [c for c in collisions if c.collision_type == "wall"]
        assert len(wall_collisions) == 1
