#!/usr/bin/env python3
"""
Demonstration of gate pass-through logic.

This script creates a simple scenario with a wall containing a gate,
and simulates drones passing through the gate and colliding with the wall.
"""

import numpy as np
from flockrl_sim import CollisionSystem, Environment, SwarmState
from flockrl_sim.environment.obstacles_types import Wall, Gate


def create_gate_demo_environment():
    """Create an environment with a wall and gate for demonstration."""
    # Wall at y=0, extending from x=-5 to x=5, height from z=-4 to z=4
    wall = Wall(
        id="demo_wall",
        type="wall",
        position=(0.0, 0.0, 0.0),
        orientation=(0.0, 0.0, 0.0),
        length=10.0,
        height=8.0,
        thickness=0.2,
        gate_ids=("demo_gate",),
    )

    # Gate at center of wall, 2m wide x 2m high
    gate = Gate(
        id="demo_gate",
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


def run_demo():
    """Run the gate pass-through demonstration."""
    print("=" * 70)
    print("Gate Pass-Through Demonstration")
    print("=" * 70)
    print()

    env = create_gate_demo_environment()
    collision_system = CollisionSystem(environment=env, drone_radius=0.5, restitution=1.0)

    print("Environment setup:")
    print("  Wall: 10m x 8m x 0.2m at position (0, 0, 0)")
    print("  Gate: 2m x 2m x 0.2m at position (0, 0, 0)")
    print("  Drone radius: 0.5m")
    print()

    # Test scenarios
    scenarios = [
        {
            "name": "Drone through center of gate",
            "positions": [[0.0, 0.0, 0.0]],
            "expected_collisions": 0,
        },
        {
            "name": "Drone near gate edge (inside)",
            "positions": [[0.9, 0.0, 0.9]],
            "expected_collisions": 0,
        },
        {
            "name": "Drone outside gate (left side)",
            "positions": [[2.0, 0.0, 0.0]],
            "expected_collisions": 1,
        },
        {
            "name": "Drone above gate",
            "positions": [[0.0, 0.0, 2.0]],
            "expected_collisions": 1,
        },
        {
            "name": "Multiple drones (mixed)",
            "positions": [
                [0.0, 0.0, 0.0],  # Through gate
                [3.0, 0.0, 0.0],  # Hitting wall
                [0.5, 0.0, 0.5],  # Through gate
                [-3.0, 0.0, 0.0],  # Hitting wall
            ],
            "expected_collisions": 2,
        },
    ]

    for scenario in scenarios:
        print("-" * 70)
        print(f"Scenario: {scenario['name']}")
        print(f"Positions: {scenario['positions']}")

        num_drones = len(scenario["positions"])
        state = SwarmState(
            pos=np.array(scenario["positions"]),
            vel=np.array([[0.0, 1.0, 0.0]] * num_drones),
            acc=np.zeros((num_drones, 3)),
            ids=np.array(range(num_drones)),
            goals=np.zeros((num_drones, 3)),  # Goals always required
            t=0.0,
        )

        _, info = collision_system(state)
        collisions = info["collisions"]
        wall_collisions = [c for c in collisions if c.collision_type == "wall"]

        print(f"Wall collisions detected: {len(wall_collisions)}")
        if wall_collisions:
            for collision in wall_collisions:
                print(f"  - Drone {collision.drone_id} collided")
                print(f"    Contact point: {collision.contact_point}")
                print(f"    Normal: {collision.normal_vector}")
                print(f"    Penetration: {collision.penetration_depth:.3f}m")

        expected = scenario["expected_collisions"]
        status = "✓ PASS" if len(wall_collisions) == expected else "✗ FAIL"
        print(
            f"Expected {expected} collision(s), got {len(wall_collisions)} - {status}"
        )
        print()

    print("=" * 70)
    print("Demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
