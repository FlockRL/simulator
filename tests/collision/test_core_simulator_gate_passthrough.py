"""
CoreSimulator integration tests for gate pass-through and collisions.
"""

import numpy as np

from flockrl_sim import CollisionSystem, Environment, SwarmState
from flockrl_sim.environment.obstacles_types import Gate, Wall
from flockrl_sim.simulator import CoreSimulator


def _make_gate_environment():
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
    return Environment(
        bounds=(-10.0, 10.0, -10.0, 10.0, -10.0, 10.0),
        obstacles=[gate, wall],
        start_position=(-5.0, -5.0, 0.0),
        goal_position=(5.0, 5.0, 0.0),
        seed=42,
    )


def _make_simulator(environment):
    collision_system = CollisionSystem(
        environment=environment, drone_radius=0.5, restitution=1.0
    )
    return CoreSimulator(
        delta_t=0.1,
        max_steps=10,
        goal_threshold=0.5,
        max_acceleration=5.0,
        terminate_on_collision=True,
        collision_system=collision_system,
        environment=environment,
        enable_frame_logging=False,
        perception_config={
            "max_range": 50.0,
            "num_rays": 8,
            "max_neighbour_range": 10.0,
        },
        reset_config={"reset_position_noise": 0.0, "reset_velocity_noise": 0.0},
    )


def test_gate_passthrough_does_not_terminate():
    """Drone centered in gate should not collide or terminate."""
    env = _make_gate_environment()
    simulator = _make_simulator(env)

    state = SwarmState(
        goals=np.array([[5.0, 5.0, 0.0]]),
        pos=np.array([[0.0, 0.0, 0.0]]),
        vel=np.zeros((1, 3)),
        acc=np.zeros((1, 3)),
        ids=np.array([0]),
        t=0.0,
    )
    simulator.start_run(initial_state=state)

    _, info = simulator.step(np.zeros((1, 3)))
    wall_collisions = [c for c in info["collisions"] if c.collision_type == "wall"]

    assert wall_collisions == []
    assert info["done"] is False
    assert info["termination_reason"] is None


def test_wall_collision_terminates():
    """Drone outside gate should collide with wall and terminate."""
    env = _make_gate_environment()
    simulator = _make_simulator(env)

    state = SwarmState(
        goals=np.array([[5.0, 5.0, 0.0]]),
        pos=np.array([[3.0, 0.0, 0.0]]),
        vel=np.zeros((1, 3)),
        acc=np.zeros((1, 3)),
        ids=np.array([0]),
        t=0.0,
    )
    simulator.start_run(initial_state=state)

    _, info = simulator.step(np.zeros((1, 3)))
    wall_collisions = [c for c in info["collisions"] if c.collision_type == "wall"]

    assert len(wall_collisions) == 1
    assert wall_collisions[0].drone_id == 0
    assert info["done"] is True
    assert info["termination_reason"] == "collision"


def test_multi_drone_gate_and_wall_collision():
    """One drone inside gate, one colliding with wall should terminate."""
    env = _make_gate_environment()
    simulator = _make_simulator(env)

    state = SwarmState(
        goals=np.array([[5.0, 5.0, 0.0], [5.0, 5.0, 0.0]]),
        pos=np.array(
            [
                [0.0, 0.0, 0.0],  # inside gate
                [3.0, 0.0, 0.0],  # wall collision
            ]
        ),
        vel=np.zeros((2, 3)),
        acc=np.zeros((2, 3)),
        ids=np.array([0, 1]),
        t=0.0,
    )
    simulator.start_run(initial_state=state)

    _, info = simulator.step(np.zeros((2, 3)))
    wall_collisions = [c for c in info["collisions"] if c.collision_type == "wall"]
    colliding_ids = {c.drone_id for c in wall_collisions}

    assert colliding_ids == {1}
    assert info["done"] is True
    assert info["termination_reason"] == "collision"
