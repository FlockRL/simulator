"""
Tests for drone-drone collision detection and response.
"""

import numpy as np

from flockrl_sim import CollisionSystem, Environment, SwarmState
from flockrl_sim.simulator import CoreSimulator


def _make_empty_environment():
    return Environment(
        bounds=(-10.0, 10.0, -10.0, 10.0, -10.0, 10.0),
        obstacles=[],
        start_position=(0.0, 0.0, 0.0),
        goal_position=(0.0, 0.0, 0.0),
        seed=0,
    )


def test_drone_collision_detected_and_symmetric():
    env = _make_empty_environment()
    collision_system = CollisionSystem(environment=env, drone_radius=0.5, restitution=1.0)

    state = SwarmState(
        goals=np.zeros((2, 3)),
        pos=np.array([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]]),
        vel=np.zeros((2, 3)),
        acc=np.zeros((2, 3)),
        ids=np.array([0, 1]),
        t=0.0,
    )

    _, info = collision_system(state)
    collisions = [c for c in info["collisions"] if c.collision_type == "drone"]

    assert len(collisions) == 2

    by_id = {c.drone_id: c for c in collisions}
    assert np.allclose(by_id[0].new_position, np.array([-0.1, 0.0, 0.0]))
    assert np.allclose(by_id[1].new_position, np.array([0.9, 0.0, 0.0]))
    assert np.allclose(by_id[0].normal_vector, np.array([-1.0, 0.0, 0.0]))
    assert np.allclose(by_id[1].normal_vector, np.array([1.0, 0.0, 0.0]))


def test_drone_collision_not_detected_when_separated():
    env = _make_empty_environment()
    collision_system = CollisionSystem(environment=env, drone_radius=0.5, restitution=1.0)

    state = SwarmState(
        goals=np.zeros((2, 3)),
        pos=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        vel=np.zeros((2, 3)),
        acc=np.zeros((2, 3)),
        ids=np.array([0, 1]),
        t=0.0,
    )

    _, info = collision_system(state)
    collisions = [c for c in info["collisions"] if c.collision_type == "drone"]

    assert collisions == []


def test_core_simulator_terminates_on_drone_collision():
    env = _make_empty_environment()
    collision_system = CollisionSystem(environment=env, drone_radius=0.5, restitution=1.0)
    simulator = CoreSimulator(
        delta_t=0.1,
        max_steps=10,
        goal_threshold=0.5,
        max_acceleration=5.0,
        terminate_on_collision=True,
        collision_system=collision_system,
        environment=env,
        enable_frame_logging=False,
        perception_config=None,
        reset_config={"reset_position_noise": 0.0, "reset_velocity_noise": 0.0},
    )

    state = SwarmState(
        goals=np.zeros((2, 3)),
        pos=np.array([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]]),
        vel=np.zeros((2, 3)),
        acc=np.zeros((2, 3)),
        ids=np.array([0, 1]),
        t=0.0,
    )
    simulator.start_run(initial_state=state)

    _, info = simulator.step(np.zeros((2, 3)))
    drone_collisions = [c for c in info["collisions"] if c.collision_type == "drone"]

    assert len(drone_collisions) == 2
    assert info["done"] is True
    assert info["termination_reason"] == "collision"
