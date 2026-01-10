import time
import numpy as np
from flockrl_sim.simulator import CoreSimulator, SwarmState

def benchmark_fps(n_drones = 1000, n_steps = 1000):
    # Creating the simulation:
    sim = CoreSimulator(
        delta_t = 1 / 60,
        collision_system = None,
        enable_logging=False,
        enable_perception=False
    )

    # Creating an initial state
    sim.state = sim.start_run(
        initial_state=SwarmState.from_initial_positions(
            positions=np.random.uniform(low = 0, high = 1, size=(n_drones, 3)), # Initializing from uniform
            ids = np.arange(0, n_drones), # Initializing n_drones
            goals=np.ones(shape = (n_drones, 3), dtype=np.float32) * 100 # End all drones at (100, 100, 100)
        )
    )

    # Starting benchmarking:
    start = time.time_ns()

    # Zero acceleration to only strain memory bandwidth:
    actions = np.zeros((n_drones, 3))

    for _ in range(n_steps):
        sim.step(actions)

    end = time.time_ns()

    # Getting the duration in seconds:
    duration = (end - start)/1e9

    fps = n_steps/duration; # Steps per second

    print(f"Throughput ({n_drones} drones): {fps:.2f} steps/sec")
    print(f"Latency per step: {1/fps*1e6:.2f} microseconds")

if __name__ == "__main__":
    benchmark_fps()