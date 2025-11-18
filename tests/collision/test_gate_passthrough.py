"""
Tests for gate pass-through collision logic.

Tests verify that drones can pass through gates embedded in walls
without triggering wall collisions, while still colliding with the
wall outside the gate volume.
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
    # Wall at y=0, extending from x=-5 to x=5, height from z=-4 to z=4
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

    # Gate at center of wall, 2m wide x 2m high
    gate = Gate(
        id="gate1",
        type="gate",
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        width=2.0,
        height=2.0,
        thickness=0.2
    )

    env = Environment(
        bounds=(-10.0, 10.0, -10.0, 10.0, -10.0, 10.0),
        obstacles=[gate, wall],  # Gate added first, then wall
        start_position=(-5.0, -5.0, 0.0),
        goal_position=(5.0, 5.0, 0.0),
        seed=42
    )

    return env


@pytest.fixture
def collision_system(simple_environment):
    """Create a collision system with the simple environment."""
    return CollisionSystem(environment=simple_environment, drone_radius=0.5)


class TestGatePassThrough:
    """Test that drones can pass through gates without colliding with walls."""

    def test_drone_inside_gate_no_collision(self, collision_system):
        """Drone centered in gate should not collide with wall."""
        # Position drone at center of gate
        state = SwarmState(
            pos=np.array([[0.0, 0.0, 0.0]]),
            vel=np.array([[1.0, 0.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        # Should have no wall collisions
        wall_collisions = [c for c in collisions if c.collision_type == "wall"]
        assert len(wall_collisions) == 0, "Drone in gate center should not collide with wall"

    def test_drone_near_gate_edge_no_collision(self, collision_system):
        """Drone near gate edge but still inside should not collide."""
        # Position drone at edge of gate (just inside)
        state = SwarmState(
            pos=np.array([[0.9, 0.0, 0.9]]),  # Near corner of 2x2 gate
            vel=np.array([[0.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        wall_collisions = [c for c in collisions if c.collision_type == "wall"]
        assert len(wall_collisions) == 0, "Drone inside gate boundary should not collide"

    def test_drone_outside_gate_does_collide(self, collision_system):
        """Drone outside gate volume should collide with wall."""
        # Position drone outside gate but intersecting wall
        # Gate is 2m wide, so x=2 is outside the gate
        state = SwarmState(
            pos=np.array([[2.0, 0.0, 0.0]]),
            vel=np.array([[0.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        wall_collisions = [c for c in collisions if c.collision_type == "wall"]
        assert len(wall_collisions) == 1, "Drone outside gate should collide with wall"
        assert wall_collisions[0].drone_id == 0

    def test_multiple_drones_mixed_positions(self, collision_system):
        """Test multiple drones, some inside gate, some outside."""
        state = SwarmState(
            pos=np.array([
                [0.0, 0.0, 0.0],    # Inside gate
                [3.0, 0.0, 0.0],    # Outside gate, colliding
                [0.5, 0.0, 0.5],    # Inside gate
                [-3.0, 0.0, 0.0],   # Outside gate, colliding
            ]),
            vel=np.array([
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]),
            acc=np.zeros((4, 3)),
            ids=np.array([0, 1, 2, 3]),
            t=0.0
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        wall_collisions = [c for c in collisions if c.collision_type == "wall"]
        colliding_ids = {c.drone_id for c in wall_collisions}

        # Only drones 1 and 3 should collide
        assert colliding_ids == {1, 3}, f"Expected drones 1,3 to collide, got {colliding_ids}"

    def test_drone_above_gate_does_collide(self, collision_system):
        """Drone above gate height should collide with wall."""
        # Gate is 2m high (from z=-1 to z=1), so z=2 is above it
        state = SwarmState(
            pos=np.array([[0.0, 0.0, 2.0]]),
            vel=np.array([[0.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        wall_collisions = [c for c in collisions if c.collision_type == "wall"]
        assert len(wall_collisions) == 1, "Drone above gate should collide with wall"

    def test_drone_below_gate_does_collide(self, collision_system):
        """Drone below gate height should collide with wall."""
        # Gate is 2m high (from z=-1 to z=1), so z=-2 is below it
        state = SwarmState(
            pos=np.array([[0.0, 0.0, -2.0]]),
            vel=np.array([[0.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        wall_collisions = [c for c in collisions if c.collision_type == "wall"]
        assert len(wall_collisions) == 1, "Drone below gate should collide with wall"


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

        # Two gates side by side
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

        # Test drones in both gates and between them
        state = SwarmState(
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

        # Only drone 2 (between gates) should collide
        assert colliding_ids == {2}, f"Expected drone 2 to collide, got {colliding_ids}"


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
            gate_ids=()  # No gates
        )

        env = Environment(
            bounds=(-10.0, 10.0, -10.0, 10.0, -10.0, 10.0),
            obstacles=[wall],
            start_position=(-5.0, -5.0, 0.0),
            goal_position=(5.0, 5.0, 0.0),
            seed=42
        )

        collision_system = CollisionSystem(environment=env, drone_radius=0.5)

        # Drone at wall center should collide
        state = SwarmState(
            pos=np.array([[0.0, 0.0, 0.0]]),
            vel=np.array([[0.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        wall_collisions = [c for c in collisions if c.collision_type == "wall"]
        assert len(wall_collisions) == 1, "Wall without gate should collide normally"


class TestOrientedGates:
    """Test gates with non-zero orientation (rotated gates)."""

    def test_rotated_gate_passthrough(self):
        """Test that rotated gates correctly allow pass-through."""
        # Wall rotated 45 degrees around z-axis
        yaw = np.pi / 4  # 45 degrees

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

        # Drone at center should not collide (inside rotated gate)
        state = SwarmState(
            pos=np.array([[0.0, 0.0, 0.0]]),
            vel=np.array([[1.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        wall_collisions = [c for c in collisions if c.collision_type == "wall"]
        # Note: This test will fail with current NotImplementedError for OBB collision
        # It's here to document expected behavior once OBB is implemented
        assert len(wall_collisions) == 0, "Drone in rotated gate should not collide"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_drone_radius_accounted_for(self):
        """Verify drone radius is considered in gate pass-through."""
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

        # Small gate: 1m x 1m
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

        # Large drone radius
        collision_system = CollisionSystem(environment=env, drone_radius=0.6)

        # Drone center at gate edge
        state = SwarmState(
            pos=np.array([[0.5, 0.0, 0.0]]),  # At edge of 1m gate
            vel=np.array([[0.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        # Drone center is inside gate, so no wall collision
        # (Note: gate pass-through checks drone center, not drone surface)
        wall_collisions = [c for c in collisions if c.collision_type == "wall"]
        assert len(wall_collisions) == 0

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
            pos=np.array([[0.0, 0.0, 0.0]]),
            vel=np.array([[0.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        wall_collisions = [c for c in collisions if c.collision_type == "wall"]
        assert len(wall_collisions) == 1, "Wall with empty gate_ids should collide"

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
            gate_ids=("nonexistent_gate",)  # Gate doesn't exist
        )

        env = Environment(
            bounds=(-10.0, 10.0, -10.0, 10.0, -10.0, 10.0),
            obstacles=[wall],  # No gate added
            start_position=(-5.0, -5.0, 0.0),
            goal_position=(5.0, 5.0, 0.0),
            seed=42
        )

        collision_system = CollisionSystem(environment=env, drone_radius=0.5)

        state = SwarmState(
            pos=np.array([[0.0, 0.0, 0.0]]),
            vel=np.array([[0.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0
        )

        # Should not crash, should treat as no gates
        _, info = collision_system(state)
        collisions = info["collisions"]

        wall_collisions = [c for c in collisions if c.collision_type == "wall"]
        assert len(wall_collisions) == 1, "Missing gate should not cause crash"
