"""Generate sample simulation data."""

import json
from pathlib import Path

import click
import numpy as np

from flockrl_sim.state import SwarmState
from flockrl_sim.simulator import SimulationFrame, SimulationRun


def generate_sample_state(
    num_drones: int = 5,
    bounds: tuple = (-5.0, 5.0, -5.0, 5.0, 0.0, 5.0),
    seed: int = 42
) -> SwarmState:
    """Generate a single sample SwarmState with random drone positions."""
    np.random.seed(seed)
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    
    positions = np.column_stack([
        np.random.uniform(x_min, x_max, num_drones),
        np.random.uniform(y_min, y_max, num_drones),
        np.random.uniform(z_min, z_max, num_drones),
    ])
    
    velocities = np.full((num_drones, 3), -1.0)
    accelerations = np.full((num_drones, 3), -1.0)
    
    return SwarmState(
        pos=positions,
        vel=velocities,
        acc=accelerations
    )


def generate_circular_trajectory(
    num_drones: int,
    num_frames: int,
    duration: float,
    bounds: tuple,
    seed: int
) -> SimulationRun:
    """Generate a complete simulation run with circular trajectories."""
    np.random.seed(seed)
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    
    # Generate random parameters for each drone
    radii = np.random.uniform(1.0, 3.0, num_drones)
    angular_speeds = np.random.uniform(0.5, 2.0, num_drones)
    center_x = np.random.uniform(x_min + 3, x_max - 3, num_drones)
    center_y = np.random.uniform(y_min + 3, y_max - 3, num_drones)
    center_z = np.random.uniform(z_min + 1, z_max - 1, num_drones)
    phase_offsets = np.random.uniform(0, 2 * np.pi, num_drones)
    
    frames = []
    dt = duration / num_frames
    
    for frame_idx in range(num_frames):
        t = frame_idx * dt
        positions = np.zeros((num_drones, 3))
        
        for i in range(num_drones):
            angle = angular_speeds[i] * t + phase_offsets[i]
            positions[i] = [
                center_x[i] + radii[i] * np.cos(angle),
                center_y[i] + radii[i] * np.sin(angle),
                center_z[i]
            ]
        
        velocities = np.full((num_drones, 3), -1.0)
        accelerations = np.full((num_drones, 3), -1.0)
        
        state = SwarmState(pos=positions, vel=velocities, acc=accelerations)
        frame = SimulationFrame(state=state, info={"timestamp": t})
        frames.append(frame)
    
    return SimulationRun(
        frames=frames,
        metadata={
            "num_drones": num_drones,
            "duration": duration,
            "num_frames": num_frames,
            "dt": dt,
            "trajectory_type": "circular",
            "seed": seed
        }
    )


def save_simulation_run(run: SimulationRun, output_path: Path) -> None:
    """Save a SimulationRun to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dict manually since it's a dataclass not pydantic
    data = {
        "metadata": run.metadata,
        "frames": [
            {
                "state": {
                    "pos": frame.state.pos.tolist(),
                    "vel": frame.state.vel.tolist(),
                    "acc": frame.state.acc.tolist()
                },
                "info": frame.info
            }
            for frame in run.frames
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


@click.command()
@click.option('--num-drones', '-n', default=5, type=int, help='Number of drones.')
@click.option('--num-frames', '-f', default=100, type=int, help='Number of frames.')
@click.option('--duration', '-d', default=10.0, type=float, help='Duration in seconds.')
@click.option('--output', '-o', default='sample_simulation.json', type=click.Path(), help='Output file.')
@click.option('--seed', '-s', default=42, type=int, help='Random seed.')
@click.option('--trajectory', type=click.Choice(['circular']), default='circular', help='Trajectory type.')
def generate(num_drones, num_frames, duration, output, seed, trajectory):
    """Generate sample simulation data."""
    bounds = (-5.0, 5.0, -5.0, 5.0, 0.0, 5.0)
    
    run = generate_circular_trajectory(
        num_drones=num_drones,
        num_frames=num_frames,
        duration=duration,
        bounds=bounds,
        seed=seed
    )
    
    output_path = Path(output)
    if output_path.suffix.lower() != '.json':
        output_path = output_path.with_suffix('.json')
    
    save_simulation_run(run, output_path)
    
    click.echo(f"✓ Generated {len(run.frames)} frames → {output_path}")

