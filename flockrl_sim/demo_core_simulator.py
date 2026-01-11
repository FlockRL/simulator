"""
Comprehensive demo script showcasing all components of the FlockRL simulator system.

This demo demonstrates:
1. Core simulator functionality (step, reset, episode management)
2. State management (SwarmState with positions, velocities, goals)
3. Environment creation (manual and from specs)
4. Collision detection and response (bounds, walls, clutter)
5. Perception system (ray-casting, neighbor detection)
6. Different obstacle types (walls, gates, rectangular prisms)
7. Episode termination conditions (success, collision, timeout)
8. Multiple episodes with reset
9. Simulation recording and saving
"""

import numpy as np
from pathlib import Path
from .simulator import CoreSimulator
from .state import SwarmState
from .environment import Environment, EnvironmentBuilder
from .environment.loader import EnvironmentSpecLoader
from .gym_env import load_config
from .environment.obstacles_types import Wall, Gate, RectangularPrism
from .collision.system import CollisionSystem
import yaml
from pathlib import Path
from .gym_env import load_config


def demo_basic_simulation():
    """Demonstrates basic simulation with multiple drones."""
    print("=" * 60)
    print("DEMO 1: Basic Multi-Drone Simulation")
    print("=" * 60)

    # Create environment
    env = Environment(
        bounds=(-10, 10, -10, 10, 0, 20),
        obstacles=[],
        start_position=(0.0, 0.0, 1.0),
        goal_position=(5.0, 5.0, 10.0),
        seed=42,
    )

    # Create simulator with collision system
    config = {
        "delta_t": 1 / 60.0,
        "max_steps": 300,
        "goal_threshold": 1.0,
        "terminate_on_collision": False,
        "max_acceleration": None,
    }

    simulator = CoreSimulator(
        delta_t=config["delta_t"],
        max_steps=config["max_steps"],
        goal_threshold=config["goal_threshold"],
        max_acceleration=config["max_acceleration"],
        terminate_on_collision=config["terminate_on_collision"],
        collision_system=CollisionSystem(env, drone_radius=1.0, restitution=1.0),
        environment=env,
    )

    # Initialize with 3 drones
    positions = np.array([[0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [0.0, 2.0, 1.0]])
    ids = np.array([0, 1, 2])
    goals = np.array([[5.0, 5.0, 10.0], [7.0, 5.0, 10.0], [5.0, 7.0, 10.0]])
    initial_state = SwarmState.from_initial_positions(positions, ids, goals)

    # Start simulation
    state = simulator.start_run(initial_state, {"demo": "basic_flight"})

    print("Starting simulation with 3 drones...")
    print(f"Initial positions:\n{state.pos}")
    print(f"Goals:\n{state.goals}")

    # Run simulation
    for step in range(300):
        # Simple circular motion actions
        t = step * simulator.delta_t
        actions = np.array(
            [
                [0.1 * np.cos(t), 0.1 * np.sin(t), 0.0],  # Drone 0: circle
                [-0.1, 0.0, 0.0],  # Drone 1: move left
                [0.0, -0.1, 0.0],  # Drone 2: move back
            ]
        )

        state, info = simulator.step(actions)

        # Check for termination
        if info["done"]:
            print(
                f"\nSimulation terminated at step {step}: {info['termination_reason']}"
            )
            break

        if step % 60 == 0:  # Print every second
            print(f"\nt={state.t:.1f}s:")
            print(f"  Positions:\n{state.pos}")
            print(
                f"  Goal distances: {[f'{d:.2f}' for d in np.linalg.norm(state.pos - state.goals, axis=1)]}"
            )

    print(f"\n✓ Episode stats: {info['episode_stats']}")
    print()


def demo_obstacles_and_collisions():
    """Demonstrates different obstacle types and collision detection."""
    print("=" * 60)
    print("DEMO 2: Obstacles and Collision Detection")
    print("=" * 60)

    # Create environment with various obstacles
    obstacles = [
        # Wall with a gate
        Wall(
            id="wall1",
            type="wall",
            position=(5.0, 0.0, 5.0),
            length=10.0,
            height=10.0,
            thickness=0.5,
            orientation=(0.0, 0.0, 0.0),
            gate_ids=["gate1"],
        ),
        Gate(
            id="gate1",
            type="gate",
            position=(5.0, 0.0, 5.0),
            width=3.0,
            height=6.0,
            thickness=0.5,
            orientation=(0.0, 0.0, 0.0),
        ),
        # Rectangular prism obstacle
        RectangularPrism(
            id="box1",
            type="RectangularPrism",
            position=(-3.0, 3.0, 2.0),
            length=2.0,
            width=2.0,
            height=4.0,
            orientation=(0.0, 0.0, 0.785),  # 45 degree rotation
            subtype="rectangular_prism",
        ),
    ]

    env = Environment(
        bounds=(-10, 10, -10, 10, 0, 20),
        obstacles=obstacles,
        start_position=(-5.0, 0.0, 5.0),
        goal_position=(10.0, 0.0, 5.0),
        seed=42,
    )

    config = {
        "delta_t": 1 / 60.0,
        "max_steps": 500,
        "goal_threshold": 1.0,
        "terminate_on_collision": True,
        "max_acceleration": None,
    }

    simulator = CoreSimulator(
        delta_t=config["delta_t"],
        max_steps=config["max_steps"],
        goal_threshold=config["goal_threshold"],
        max_acceleration=config["max_acceleration"],
        terminate_on_collision=config["terminate_on_collision"],
        collision_system=CollisionSystem(env, drone_radius=0.5, restitution=1.0),
        environment=env,
    )

    # Single drone trying to navigate through gate
    positions = np.array([[-5.0, 0.0, 5.0]])
    ids = np.array([0])
    goals = np.array([[10.0, 0.0, 5.0]])
    initial_state = SwarmState.from_initial_positions(positions, ids, goals)

    state = simulator.start_run(initial_state, {"demo": "obstacle_navigation"})

    print(f"Environment has {len(env.obstacles)} obstacles:")
    for obs in env.obstacles:
        print(f"  - {obs.type} (id: {obs.id})")
    print(f"\nDrone starting at: {state.pos[0]}")
    print(f"Goal at: {state.goals[0]}")
    print("\nNavigating towards goal...")

    for step in range(500):
        # Simple proportional controller towards goal
        direction = state.goals[0] - state.pos[0]
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        actions = direction.reshape(1, 3) * 2.0

        state, info = simulator.step(actions)

        if info["done"]:
            print(
                f"\n✓ Episode terminated at step {step}: {info['termination_reason']}"
            )
            print(f"  Final position: {state.pos[0]}")
            print(f"  Collisions: {info['episode_stats']['collision_count']}")
            if info.get("collisions"):
                for col in info["collisions"]:
                    print(
                        f"    - Collision with {col.collision_type} (drone {col.drone_id})"
                    )
            break

        if step % 60 == 0:
            goal_dist = np.linalg.norm(state.pos[0] - state.goals[0])
            print(f"  Step {step}: position={state.pos[0]}, goal_dist={goal_dist:.2f}m")

    print()


def demo_perception_system():
    """Demonstrates perception system with ray-casting and neighbor detection."""
    print("=" * 60)
    print("DEMO 3: Perception System (Ray-Casting & Neighbors)")
    print("=" * 60)

    # Create environment with obstacles
    obstacles = [
        Wall(
            id="wall1",
            type="wall",
            position=(5.0, 0.0, 5.0),
            length=10.0,
            height=10.0,
            thickness=0.5,
            orientation=(0.0, 0.0, 0.0),
            gate_ids=[],
        ),
        RectangularPrism(
            id="box1",
            type="RectangularPrism",
            position=(-2.0, 2.0, 3.0),
            length=1.5,
            width=1.5,
            height=3.0,
            orientation=(0.0, 0.0, 0.0),
            subtype="rectangular_prism",
        ),
    ]

    env = Environment(
        bounds=(-10, 10, -10, 10, 0, 20),
        obstacles=obstacles,
        start_position=(0.0, 0.0, 5.0),
        goal_position=(10.0, 0.0, 5.0),
        seed=42,
    )

    # This demo doesn't use config, so we'll use defaults from config.yml
    # For demo purposes, we'll create a minimal config
    simulator = CoreSimulator(
        delta_t=1 / 60.0,
        max_steps=1000,
        goal_threshold=0.5,
        max_acceleration=None,
        terminate_on_collision=True,
        collision_system=CollisionSystem(env, drone_radius=1.0, restitution=1.0),
        environment=env,
    )

    # Multiple drones for neighbor detection
    positions = np.array([[0.0, 0.0, 5.0], [2.0, 0.0, 5.0], [0.0, 2.0, 5.0]])
    ids = np.array([0, 1, 2])
    goals = np.array([[10.0, 0.0, 5.0], [10.0, 2.0, 5.0], [10.0, -2.0, 5.0]])
    initial_state = SwarmState.from_initial_positions(positions, ids, goals)

    state = simulator.start_run(initial_state, {"demo": "perception"})

    print("Testing perception system with 3 drones...")

    # Step once and examine observations
    actions = np.zeros((3, 3))
    state, info = simulator.step(actions)

    observations = info.get("observations", [])
    print(f"\n✓ Generated {len(observations)} observations (one per drone)")

    for i, obs in enumerate(observations):
        print(f"\nDrone {i} observations:")
        print(
            f"  - Ray-cast ranges: {obs.ranges.shape} (min={obs.ranges.min():.2f}m, max={obs.ranges.max():.2f}m)"
        )
        print(f"  - Obstacle hits: {np.sum(obs.hits)}/{len(obs.hits)} rays")
        print(f"  - Neighbor vectors: {obs.neighbor_vectors.shape}")
        if len(obs.neighbor_vectors) > 0:
            print(f"    (Detected {len(obs.neighbor_vectors)} neighbors)")

    print()


def demo_environment_loading():
    """Demonstrates loading environments from JSON specs."""
    print("=" * 60)
    print("DEMO 4: Environment Loading from Specs")
    print("=" * 60)

    loader = EnvironmentSpecLoader()

    print("Available presets:", loader.list_presets())

    # Load a preset
    spec = loader.load("simple")
    print("\n✓ Loaded 'simple' preset")

    # Load config for EnvironmentBuilder parameters
    config = load_config()
    env_config = config["environment"]
    spawn_clearance = env_config["spawn_clearance"]
    max_placement_attempts = env_config["max_placement_attempts"]

    # Build environment from spec
    builder = EnvironmentBuilder.from_spec(spec, spawn_clearance, max_placement_attempts)
    env = builder.config

    print("✓ Built environment:")
    print(f"  - Bounds: {env.bounds}")
    print(f"  - Obstacles: {len(env.obstacles)}")
    print(f"  - Start: {env.start_position}")
    print(f"  - Goal: {env.goal_position}")

    # Run a quick simulation
    config = {
        "delta_t": 1 / 60.0,
        "max_steps": 100,
        "goal_threshold": 0.5,
        "terminate_on_collision": True,
        "max_acceleration": None,
    }
    simulator = CoreSimulator(
        delta_t=config["delta_t"],
        max_steps=config["max_steps"],
        goal_threshold=config["goal_threshold"],
        max_acceleration=config["max_acceleration"],
        terminate_on_collision=config["terminate_on_collision"],
        collision_system=CollisionSystem(env, drone_radius=1.0, restitution=1.0),
        environment=env,
    )

    state = simulator.start_run()
    print(f"\n✓ Simulator initialized with {state.pos.shape[0]} drone(s)")
    print()


def demo_episode_management():
    """Demonstrates episode reset and multiple episodes."""
    print("=" * 60)
    print("DEMO 5: Episode Management (Reset & Multiple Episodes)")
    print("=" * 60)

    env = Environment(
        bounds=(-10, 10, -10, 10, 0, 20),
        obstacles=[],
        start_position=(0.0, 0.0, 2.0),
        goal_position=(8.0, 8.0, 10.0),
        seed=42,
    )

    config = {
        "delta_t": 1 / 60.0,
        "max_steps": 200,
        "goal_threshold": 1.0,
        "terminate_on_collision": False,
        "max_acceleration": None,
    }

    simulator = CoreSimulator(
        delta_t=config["delta_t"],
        max_steps=config["max_steps"],
        goal_threshold=config["goal_threshold"],
        max_acceleration=config["max_acceleration"],
        terminate_on_collision=config["terminate_on_collision"],
        collision_system=CollisionSystem(env, drone_radius=1.0, restitution=1.0),
        environment=env,
    )

    # Run 3 episodes
    for ep in range(3):
        print(f"\n--- Episode {ep + 1} ---")

        # First episode: start_run, subsequent: reset
        if ep == 0:
            state = simulator.start_run()
        else:
            state = simulator.reset(randomize=True, seed=ep)
            print(f"✓ Reset with randomization (seed={ep})")

        print(f"Initial position: {state.pos[0]}")
        print(f"Goal: {state.goals[0]}")

        # Simple controller
        for step in range(200):
            direction = state.goals[0] - state.pos[0]
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            actions = direction.reshape(1, 3) * 2.0

            state, info = simulator.step(actions)

            if info["done"]:
                print(f"✓ Terminated: {info['termination_reason']}")
                print(f"  Steps: {info['episode_stats']['total_steps']}")
                print(
                    f"  Final goal distance: {info['episode_stats']['final_goal_distance']:.2f}m"
                )
                break

    print()


def demo_termination_conditions():
    """Demonstrates different episode termination conditions."""
    print("=" * 60)
    print("DEMO 6: Episode Termination Conditions")
    print("=" * 60)

    # Test 1: Success (reaching goal)
    print("\n1. Testing SUCCESS termination (reaching goal)...")
    env = Environment(
        bounds=(-10, 10, -10, 10, 0, 20),
        obstacles=[],
        start_position=(0.0, 0.0, 2.0),
        goal_position=(2.0, 0.0, 2.0),  # Close goal
        seed=42,
    )

    config = {
        "delta_t": 1 / 60.0,
        "max_steps": 1000,
        "goal_threshold": 0.5,  # Large threshold for easy success
        "terminate_on_collision": False,
        "max_acceleration": None,
    }

    simulator = CoreSimulator(
        delta_t=config["delta_t"],
        max_steps=config["max_steps"],
        goal_threshold=config["goal_threshold"],
        max_acceleration=config["max_acceleration"],
        terminate_on_collision=config["terminate_on_collision"],
        collision_system=CollisionSystem(env, drone_radius=1.0, restitution=1.0),
        environment=env,
    )

    state = simulator.start_run()
    for step in range(100):
        direction = state.goals[0] - state.pos[0]
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        actions = direction.reshape(1, 3) * 5.0

        state, info = simulator.step(actions)
        if info["done"]:
            print(f"  ✓ Success! Terminated: {info['termination_reason']}")
            break

    # Test 2: Collision
    print("\n2. Testing COLLISION termination...")
    obstacles = [
        RectangularPrism(
            id="box1",
            type="RectangularPrism",
            position=(3.0, 0.0, 2.0),
            length=2.0,
            width=2.0,
            height=4.0,
            orientation=(0.0, 0.0, 0.0),
            subtype="rectangular_prism",
        )
    ]

    env = Environment(
        bounds=(-10, 10, -10, 10, 0, 20),
        obstacles=obstacles,
        start_position=(0.0, 0.0, 2.0),
        goal_position=(10.0, 0.0, 2.0),
        seed=42,
    )

    simulator = CoreSimulator(
        delta_t=config["delta_t"],
        max_steps=config["max_steps"],
        goal_threshold=config["goal_threshold"],
        max_acceleration=config["max_acceleration"],
        terminate_on_collision=True,
        collision_system=CollisionSystem(env, drone_radius=0.5, restitution=1.0),
        environment=env,
    )

    state = simulator.start_run()
    for step in range(100):
        actions = np.array([[5.0, 0.0, 0.0]])  # Move directly into obstacle
        state, info = simulator.step(actions)
        if info["done"]:
            print(f"  ✓ Collision detected! Terminated: {info['termination_reason']}")
            print(f"    Collisions: {info['episode_stats']['collision_count']}")
            break

    # Test 3: Timeout
    print("\n3. Testing TIMEOUT termination...")
    env = Environment(
        bounds=(-10, 10, -10, 10, 0, 20),
        obstacles=[],
        start_position=(0.0, 0.0, 2.0),
        goal_position=(100.0, 100.0, 100.0),  # Unreachable goal
        seed=42,
    )

    simulator = CoreSimulator(
        delta_t=config["delta_t"],
        max_steps=50,
        goal_threshold=config["goal_threshold"],
        max_acceleration=config["max_acceleration"],
        terminate_on_collision=False,
        collision_system=CollisionSystem(env, drone_radius=1.0, restitution=1.0),
        environment=env,
    )

    state = simulator.start_run()
    for step in range(100):
        actions = np.array([[0.1, 0.1, 0.1]])
        state, info = simulator.step(actions)
        if info["done"]:
            print(f"  ✓ Timeout! Terminated: {info['termination_reason']}")
            print(f"    Steps: {info['episode_stats']['total_steps']}")
            break

    print()


def demo_save_and_record():
    """Demonstrates simulation recording and saving."""
    print("=" * 60)
    print("DEMO 7: Simulation Recording and Saving")
    print("=" * 60)

    env = Environment(
        bounds=(-10, 10, -10, 10, 0, 20),
        obstacles=[],
        start_position=(0.0, 0.0, 1.0),
        goal_position=(5.0, 5.0, 10.0),
        seed=42,
    )

    simulator = CoreSimulator(
        delta_t=1 / 60.0,
        max_steps=100,
        goal_threshold=0.5,
        max_acceleration=None,
        terminate_on_collision=True,
        collision_system=CollisionSystem(env, drone_radius=1.0, restitution=1.0),
        environment=env,
    )

    positions = np.array([[0.0, 0.0, 1.0]])
    ids = np.array([0])
    goals = np.array([[5.0, 5.0, 10.0]])
    initial_state = SwarmState.from_initial_positions(positions, ids, goals)

    state = simulator.start_run(
        initial_state, {"demo": "recording", "description": "Test simulation recording"}
    )

    # Run short simulation
    for step in range(60):
        t = step * simulator.delta_t
        actions = np.array([[0.1 * np.cos(t), 0.1 * np.sin(t), 0.0]])
        state, info = simulator.step(actions)

    # Save to file
    output_path = Path("demo_simulation_output.json")
    simulator.save_run(output_path)

    print(f"✓ Simulation recorded and saved to {output_path}")
    print(f"  Total frames: {len(simulator.current_run.frames)}")
    print(f"  Metadata: {simulator.current_run.metadata}")
    print(f"  First frame time: {simulator.current_run.frames[0].state.t:.3f}s")
    print(f"  Last frame time: {simulator.current_run.frames[-1].state.t:.3f}s")
    print()


def main():
    """Run all demos to showcase complete system functionality."""
    print("\n" + "=" * 60)
    print("FlockRL Simulator - Comprehensive Demo")
    print("=" * 60)
    print("\nThis demo showcases all components of the simulator system:\n")

    try:
        demo_basic_simulation()
        demo_obstacles_and_collisions()
        demo_perception_system()
        demo_environment_loading()
        demo_episode_management()
        demo_termination_conditions()
        demo_save_and_record()

        print("=" * 60)
        print("✓ All demos completed successfully!")
        print("=" * 60)
        print("\nSystem components demonstrated:")
        print("  ✓ Core simulator (step, reset, episode management)")
        print("  ✓ State management (SwarmState with goals)")
        print("  ✓ Environment creation (manual and from specs)")
        print("  ✓ Collision detection (bounds, walls, clutter, gates)")
        print("  ✓ Perception system (ray-casting, neighbor detection)")
        print("  ✓ Obstacle types (walls, gates, rectangular prisms)")
        print("  ✓ Episode termination (success, collision, timeout)")
        print("  ✓ Multiple episodes with reset")
        print("  ✓ Simulation recording and saving")
        print()

    except Exception as e:
        print(f"\n✗ Demo failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
