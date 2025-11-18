import numpy as np
from pathlib import Path
from flockrl_sim import CoreSimulator, SwarmState

def demo_basic_simulation():
    # Create simulator
    simulator = CoreSimulator(delta_t=1/60.0)
    
    # Initialize with 3 drones
    positions = np.array([
        [0.0, 0.0, 1.0],
        [2.0, 0.0, 1.0], 
        [0.0, 2.0, 1.0]
    ])
    ids = np.array([0, 1, 2])
    initial_state = SwarmState.from_initial_positions(positions, ids)
    
    # Start simulation
    simulator.start_run(initial_state, {"demo": "basic_flight"})
    
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
        simulator.log_frame(info)
        
        if step % 60 == 0:  # Print every second
            print(f"t={state.t:.1f}s: positions=\n{state.pos}")
    
    # Save results
    simulator.save_run(Path("demo_output.json"))
    print("✓ Simulation saved to demo_output.json")

if __name__ == "__main__":
    demo_basic_simulation()