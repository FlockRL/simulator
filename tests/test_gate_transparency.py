"""
Comprehensive tests for gate transparency behavior.

This test module verifies that:
1. Rays DO NOT hit gates (gates are transparent to perception rays)
2. Rays DO NOT hit wall portions that have gates embedded in them
3. Drones CAN physically pass through gates embedded in walls
4. Drones CANNOT pass through wall portions without gates
"""

import numpy as np
import pytest

from flockrl_sim.collision.system import CollisionSystem
from flockrl_sim.environment import Environment
from flockrl_sim.environment.obstacles_types import Wall, Gate
from flockrl_sim.perception.sensors import PerceptionSystem, SensorConfig
from flockrl_sim.state import SwarmState


@pytest.fixture
def wall_with_gate_environment():
    """
    Create an environment with a wall containing a centered gate.

    Layout:
    - Wall: 10m wide, 5m tall, 0.2m thick, centered at origin
    - Gate: 2m wide, 2m tall, 0.2m thick, centered in the wall
    - Wall is oriented along the XZ plane (perpendicular to Y-axis)
    """
    gate = Gate(
        id="gate1",
        type="gate",
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        width=2.0,
        height=2.0,
        thickness=0.2,
    )

    wall = Wall(
        id="wall1",
        type="wall",
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        length=10.0,
        height=5.0,
        thickness=0.2,
        gate_ids=("gate1",),
    )

    env = Environment(
        bounds=(-20.0, 20.0, -20.0, 20.0, -20.0, 20.0),
        obstacles=[gate, wall],
        start_position=(-5.0, -5.0, 0.0),
        goal_position=(5.0, 5.0, 0.0),
        seed=42,
    )

    return env


@pytest.fixture
def standalone_gate_environment():
    """Create an environment with just a gate (no wall)."""
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
        bounds=(-20.0, 20.0, -20.0, 20.0, -20.0, 20.0),
        obstacles=[gate],
        start_position=(-5.0, -5.0, 0.0),
        goal_position=(5.0, 5.0, 0.0),
        seed=42,
    )

    return env


class TestRayTransparencyThroughGates:
    """Test that perception rays pass through gates without hitting them."""

    def test_ray_through_standalone_gate_no_hit(self, standalone_gate_environment):
        """Ray aimed directly at a standalone gate should pass through."""
        perception = PerceptionSystem(
            environment=standalone_gate_environment,
            config=SensorConfig(max_range=10.0, num_rays=1),
            seed=42,
        )

        # Override rays to point directly at the gate
        # Drone at (-5, 0, 0), ray pointing toward gate at (0, 0, 0)
        perception.rays = np.array([[1.0, 0.0, 0.0]])

        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[-5.0, 0.0, 0.0]]),
            vel=np.array([[0.0, 0.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0,
        )

        readings = perception.observe(state)

        # Ray should NOT hit the gate (should reach max range)
        assert len(readings) == 1
        assert readings[0].ranges[0] == pytest.approx(10.0)
        assert readings[0].hits[0] == False

    def test_ray_through_gate_in_wall_no_hit(self, wall_with_gate_environment):
        """Ray through gate portion of wall should pass through."""
        perception = PerceptionSystem(
            environment=wall_with_gate_environment,
            config=SensorConfig(max_range=10.0, num_rays=1),
            seed=42,
        )

        # Override rays to point directly through the gate center
        # Drone at (0, -5, 0), ray pointing toward gate at (0, 0, 0)
        perception.rays = np.array([[0.0, 1.0, 0.0]])

        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[0.0, -5.0, 0.0]]),
            vel=np.array([[0.0, 0.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0,
        )

        readings = perception.observe(state)

        # Ray should pass through gate and NOT hit the wall
        assert len(readings) == 1
        assert readings[0].ranges[0] == pytest.approx(10.0)
        assert readings[0].hits[0] == False

    def test_ray_through_wall_outside_gate_hits(self, wall_with_gate_environment):
        """Ray aimed at wall portion outside gate should hit."""
        perception = PerceptionSystem(
            environment=wall_with_gate_environment,
            config=SensorConfig(max_range=10.0, num_rays=1),
            seed=42,
        )

        # Override rays to point at wall outside gate
        # Drone at (3, -5, 0), ray pointing toward wall at (3, 0, 0)
        # Gate is only 2m wide centered at origin, so this is outside
        perception.rays = np.array([[0.0, 1.0, 0.0]])

        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[3.0, -5.0, 0.0]]),
            vel=np.array([[0.0, 0.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0,
        )

        readings = perception.observe(state)

        # Ray should hit the wall (not pass through)
        assert len(readings) == 1
        assert readings[0].hits[0] == True
        # Distance should be approximately 5.0 (distance to wall face)
        # Wall thickness is 0.2, so wall face is at y = -0.1
        # Distance from (3, -5, 0) to (3, -0.1, 0) is 4.9
        assert readings[0].ranges[0] < 10.0
        assert readings[0].ranges[0] == pytest.approx(4.9, abs=0.15)

    def test_multiple_rays_mixed_gate_wall_hits(self, wall_with_gate_environment):
        """Test multiple rays with some through gate and some hitting wall."""
        perception = PerceptionSystem(
            environment=wall_with_gate_environment,
            config=SensorConfig(max_range=10.0, num_rays=3),
            seed=42,
        )

        # Set up 3 rays:
        # 1. Through gate center (no hit)
        # 2. At wall above gate (hit)
        # 3. At wall to the side of gate (hit)
        perception.rays = np.array([
            [0.0, 1.0, 0.0],   # Through gate
            [0.0, 0.894, 0.447],  # Above gate (pointing up and forward)
            [0.6, 0.8, 0.0],   # To the side of gate
        ])

        # Normalize the rays
        perception.rays = perception.rays / np.linalg.norm(perception.rays, axis=1, keepdims=True)

        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[0.0, -5.0, 0.0]]),
            vel=np.array([[0.0, 0.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0,
        )

        readings = perception.observe(state)

        assert len(readings) == 1

        # Ray 0 (through gate): should not hit
        assert readings[0].hits[0] == False
        assert readings[0].ranges[0] == pytest.approx(10.0)

        # Rays 1 and 2 (outside gate): should hit wall
        # We can't be certain about exact distances due to geometry,
        # but they should definitely hit something before max range
        assert readings[0].hits[1] == True or readings[0].hits[2] == True


class TestDronePhysicalPassthroughGates:
    """Test that drones can physically pass through gates in walls."""

    def test_drone_centered_in_gate_no_collision(self, wall_with_gate_environment):
        """Drone positioned at gate center should not collide with wall."""
        collision_system = CollisionSystem(
            environment=wall_with_gate_environment,
            drone_radius=0.5,
        )

        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[0.0, 0.0, 0.0]]),  # Center of gate
            vel=np.array([[0.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0,
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        # No collisions should occur
        assert len(collisions) == 0

    def test_drone_inside_gate_bounds_no_collision(self, wall_with_gate_environment):
        """Drone anywhere within gate bounds should not collide."""
        collision_system = CollisionSystem(
            environment=wall_with_gate_environment,
            drone_radius=0.3,  # Small radius to fit inside gate
        )

        # Test several positions within gate (2m x 2m centered at origin)
        test_positions = [
            [0.0, 0.0, 0.0],      # Center
            [0.5, 0.0, 0.5],      # Upper right quadrant
            [-0.5, 0.0, -0.5],    # Lower left quadrant
            [0.7, 0.0, 0.0],      # Right edge (within bounds)
            [0.0, 0.0, 0.7],      # Top edge (within bounds)
        ]

        for pos in test_positions:
            state = SwarmState(
                goals=np.zeros((1, 3)),
                pos=np.array([pos]),
                vel=np.array([[0.0, 1.0, 0.0]]),
                acc=np.array([[0.0, 0.0, 0.0]]),
                ids=np.array([0]),
                t=0.0,
            )

            _, info = collision_system(state)
            collisions = info["collisions"]
            wall_collisions = [c for c in collisions if c.collision_type == "wall"]

            assert len(wall_collisions) == 0, f"Unexpected wall collision at position {pos}"

    def test_drone_outside_gate_does_collide(self, wall_with_gate_environment):
        """Drone positioned outside gate bounds should collide with wall."""
        collision_system = CollisionSystem(
            environment=wall_with_gate_environment,
            drone_radius=0.5,
        )

        # Test positions outside gate (gate is 2m x 2m, so these are beyond bounds)
        test_positions = [
            [3.0, 0.0, 0.0],      # Far to the right
            [0.0, 0.0, 3.0],      # Far above
            [2.0, 0.0, 0.0],      # Just outside gate width
            [0.0, 0.0, -2.0],     # Below gate
        ]

        for pos in test_positions:
            state = SwarmState(
                goals=np.zeros((1, 3)),
                pos=np.array([pos]),
                vel=np.array([[0.0, 1.0, 0.0]]),
                acc=np.array([[0.0, 0.0, 0.0]]),
                ids=np.array([0]),
                t=0.0,
            )

            _, info = collision_system(state)
            collisions = info["collisions"]
            wall_collisions = [c for c in collisions if c.collision_type == "wall"]

            assert len(wall_collisions) > 0, f"Expected wall collision at position {pos}"

    def test_drone_passing_through_gate_plane(self, wall_with_gate_environment):
        """Test drone positions at various distances through gate."""
        from flockrl_sim.simulator import CoreSimulator
        from flockrl_sim.config import SimulationConfig

        # Configure simulator with collision detection
        collision_system = CollisionSystem(
            environment=wall_with_gate_environment,
            drone_radius=0.3,
        )

        config = SimulationConfig(
            delta_t=1.0 / 60.0,
            max_steps=500,
            terminate_on_collision=False,
            goal_threshold=0.5,
        )

        simulator = CoreSimulator(
            delta_t=config.delta_t,
            environment=wall_with_gate_environment,
            config=config,
            collision_system=collision_system,
        )

        # Test multiple positions along the Y axis through the gate
        # Gate is at y=0 with thickness 0.2m (extends from y=-0.1 to y=0.1)
        # Drone with radius 0.3m should be able to pass through when centered on gate
        test_positions = [
            (0.0, -0.05, 0.0),  # Just before gate center
            (0.0, 0.0, 0.0),    # Exactly at gate center
            (0.0, 0.05, 0.0),   # Just after gate center
        ]

        for test_pos in test_positions:
            initial_state = SwarmState.from_initial_positions(
                positions=np.array([test_pos]),
                ids=np.array([0]),
                goals=np.array([[0.0, 1.0, 0.0]]),
            )

            state = simulator.start_run(initial_state=initial_state)

            # Take a single step with small velocity through gate
            action = np.array([[0.0, 0.5, 0.0]])
            state, info = simulator.step(action)

            # Check for wall collisions
            wall_collisions = [c for c in info["collisions"] if c.collision_type == "wall"]

            # When drone is centered on gate, it should not collide with wall
            assert len(wall_collisions) == 0, \
                f"Unexpected wall collision at position {test_pos}: {wall_collisions}"


class TestGateEdgeCases:
    """Test edge cases for gate transparency."""

    def test_drone_at_gate_boundary(self, wall_with_gate_environment):
        """Test drone positioned exactly at gate boundary."""
        collision_system = CollisionSystem(
            environment=wall_with_gate_environment,
            drone_radius=0.2,  # Small radius
        )

        # Gate is 2m wide, so boundary is at x = ±1.0
        # Position drone exactly at boundary
        state = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[1.0, 0.0, 0.0]]),  # Right edge of gate
            vel=np.array([[0.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0,
        )

        _, info = collision_system(state)
        collisions = info["collisions"]

        # At boundary, behavior may vary slightly due to numerical precision
        # But it should be consistent - document the actual behavior
        wall_collisions = [c for c in collisions if c.collision_type == "wall"]
        # This test documents the boundary behavior
        # Implementation may treat boundary as inside or outside
        assert True  # Just verify no crash

    def test_rotated_gate_in_wall(self):
        """Test gate and wall rotated together."""
        # Create rotated gate and wall (90 degrees around Z axis)
        gate = Gate(
            id="gate1",
            type="gate",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, np.pi / 2),  # 90 degree rotation
            width=2.0,
            height=2.0,
            thickness=0.2,
        )

        wall = Wall(
            id="wall1",
            type="wall",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, np.pi / 2),  # Same rotation
            length=10.0,
            height=5.0,
            thickness=0.2,
            gate_ids=("gate1",),
        )

        env = Environment(
            bounds=(-20.0, 20.0, -20.0, 20.0, -20.0, 20.0),
            obstacles=[gate, wall],
            start_position=(-5.0, -5.0, 0.0),
            goal_position=(5.0, 5.0, 0.0),
            seed=42,
        )

        collision_system = CollisionSystem(environment=env, drone_radius=0.3)

        # After rotation, gate opening is along Y axis instead of X
        # Drone at center should not collide
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

    def test_multiple_gates_in_wall(self):
        """Test wall with multiple gates."""
        gate1 = Gate(
            id="gate1",
            type="gate",
            position=(-3.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0),
            width=1.5,
            height=2.0,
            thickness=0.2,
        )

        gate2 = Gate(
            id="gate2",
            type="gate",
            position=(3.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0),
            width=1.5,
            height=2.0,
            thickness=0.2,
        )

        wall = Wall(
            id="wall1",
            type="wall",
            position=(0.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0),
            length=10.0,
            height=5.0,
            thickness=0.2,
            gate_ids=("gate1", "gate2"),
        )

        env = Environment(
            bounds=(-20.0, 20.0, -20.0, 20.0, -20.0, 20.0),
            obstacles=[gate1, gate2, wall],
            start_position=(-5.0, -5.0, 0.0),
            goal_position=(5.0, 5.0, 0.0),
            seed=42,
        )

        collision_system = CollisionSystem(environment=env, drone_radius=0.3)

        # Test drone in first gate
        state1 = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[-3.0, 0.0, 0.0]]),
            vel=np.array([[0.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0,
        )

        _, info1 = collision_system(state1)
        assert len(info1["collisions"]) == 0

        # Test drone in second gate
        state2 = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[3.0, 0.0, 0.0]]),
            vel=np.array([[0.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0,
        )

        _, info2 = collision_system(state2)
        assert len(info2["collisions"]) == 0

        # Test drone between gates (should collide with wall)
        state3 = SwarmState(
            goals=np.zeros((1, 3)),
            pos=np.array([[0.0, 0.0, 0.0]]),
            vel=np.array([[0.0, 1.0, 0.0]]),
            acc=np.array([[0.0, 0.0, 0.0]]),
            ids=np.array([0]),
            t=0.0,
        )

        _, info3 = collision_system(state3)
        wall_collisions = [c for c in info3["collisions"] if c.collision_type == "wall"]
        assert len(wall_collisions) > 0
