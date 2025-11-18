import numpy as np
from pathlib import Path
from .simulator import CoreSimulator
from .state import SwarmState
from .environment import Environment
from .collision.system import CollisionSystem
from .config import SimulationConfig

def demo_basic_simulation():
    """Demonstrates basic simulation with multiple drones."""
    # Create environment
    env = Environment(
        bounds=(-10, 10, -10, 10, 0, 20),
        obstacles=[],
        start_position=(0.0, 0.0, 1.0),
        goal_position=(5.0, 5.0, 10.0),
        seed=42
    )
    
    # Create simulator with collision system
    config = SimulationConfig(
        delta_t=1/60.0,
        max_steps=300,
        goal_threshold=1.0,
        terminate_on_collision=False
    )
    
    simulator = CoreSimulator(
        delta_t=config.delta_t,
        collision_system=CollisionSystem(env),
        environment=env,
        config=config
    )
    
    # Initialize with 3 drones
    positions = np.array([
        [0.0, 0.0, 1.0],
        [2.0, 0.0, 1.0], 
        [0.0, 2.0, 1.0]
    ])
    ids = np.array([0, 1, 2])
    goals = np.array([
        [5.0, 5.0, 10.0],
        [7.0, 5.0, 10.0],
        [5.0, 7.0, 10.0]
    ])
    initial_state = SwarmState.from_initial_positions(positions, ids, goals)
    
    # Start simulation
    state = simulator.start_run(initial_state, {"demo": "basic_flight"})
    
    print("Starting simulation with 3 drones...")
    print(f"Initial positions:\n{state.pos}")
    print(f"Goals:\n{state.goals}")
    
    # Run simulation for 300 steps (5 seconds at 60 Hz)
    for step in range(300):
        # Simple circular motion actions
        t = step * simulator.delta_t
        actions = np.array([
            [0.1 * np.cos(t), 0.1 * np.sin(t), 0.0],  # Drone 0: circle
            [-0.1, 0.0, 0.0],                          # Drone 1: move left
            [0.0, -0.1, 0.0]                           # Drone 2: move back
        ])
        
        state, info = simulator.step(actions)
        
        # Check for termination
        if info["done"]:
            print(f"\nSimulation terminated at step {step}: {info['termination_reason']}")
            break
        
        if step % 60 == 0:  # Print every second
            print(f"\nt={state.t:.1f}s:")
            print(f"  Positions:\n{state.pos}")
            print(f"  Goal distances: {[f'{d:.2f}' for d in np.linalg.norm(state.pos - state.goals, axis=1)]}")
    
    # Save results
    output_path = Path("demo_simulation_output.json")
    simulator.save_run(output_path)
    print(f"\n✓ Simulation saved to {output_path}")
    print(f"  Total frames: {len(simulator.current_run.frames)}")
    print(f"  Episode stats: {info['episode_stats']}")

if __name__ == "__main__":
    demo_basic_simulation()