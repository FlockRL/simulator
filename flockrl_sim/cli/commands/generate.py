import json
from pathlib import Path
import click
import numpy as np
from flockrl_sim.state import SwarmState
from flockrl_sim.simulator import SimulationFrame, SimulationRun
from flockrl_sim.environment import EnvironmentSpecLoader

def generate_sample_state(num_drones: int = 5, bounds: tuple = (-5.0, 5.0, -5.0, 5.0, 0.0, 5.0),
                         seed: int = 42) -> SwarmState:
    np.random.seed(seed)
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    positions = np.column_stack([
        np.random.uniform(x_min, x_max, num_drones),
        np.random.uniform(y_min, y_max, num_drones),
        np.random.uniform(z_min, z_max, num_drones)
    ])
    return SwarmState(pos=positions, vel=np.full((num_drones, 3), -1.0),
                     acc=np.full((num_drones, 3), -1.0))


def generate_circular_trajectory(num_drones: int, num_frames: int, duration: float,
                                bounds: tuple, seed: int) -> SimulationRun:
    np.random.seed(seed)
    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    radii = np.random.uniform(1.0, 3.0, num_drones)
    angular_speeds = np.random.uniform(0.5, 2.0, num_drones)
    center_x = np.random.uniform(x_min + 3, x_max - 3, num_drones)
    center_y = np.random.uniform(y_min + 3, y_max - 3, num_drones)
    center_z = np.random.uniform(z_min + 1, z_max - 1, num_drones)
    phase_offsets = np.random.uniform(0, 2 * np.pi, num_drones)
    dt = duration / num_frames

    frames = []
    for frame_idx in range(num_frames):
        t = frame_idx * dt
        positions = np.array([
            [center_x[i] + radii[i] * np.cos(angular_speeds[i] * t + phase_offsets[i]),
             center_y[i] + radii[i] * np.sin(angular_speeds[i] * t + phase_offsets[i]),
             center_z[i]]
            for i in range(num_drones)
        ])
        state = SwarmState(pos=positions, vel=np.full((num_drones, 3), -1.0),
                          acc=np.full((num_drones, 3), -1.0))
        frames.append(SimulationFrame(state=state, info={"timestamp": t}))

    return SimulationRun(
        frames=frames,
        metadata={
            "num_drones": num_drones,
            "duration": duration,
            "num_frames": num_frames,
            "dt": dt,
            "trajectory_type": "circular",
            "seed": seed,
            "bounds": bounds
        }
    )


def save_simulation_run(run: SimulationRun, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "metadata": run.metadata,
        "frames": [{
            "state": {
                "pos": f.state.pos.tolist(),
                "vel": f.state.vel.tolist(),
                "acc": f.state.acc.tolist()
            },
            "info": f.info
        } for f in run.frames]
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
@click.option('--environment', '-e', default='simple', type=str, help='Environment spec (preset name or path to JSON file).')
def generate(num_drones, num_frames, duration, output, seed, trajectory, environment):
    try:
        spec = EnvironmentSpecLoader().load(environment)
        bounds = spec.bounds
        click.echo(f"Using environment: {spec.name}")
        click.echo(f"Bounds: {bounds}")
    except Exception as e:
        click.echo(f"Warning: Could not load environment '{environment}': {e}", err=True)
        click.echo("Using default bounds: (-5.0, 5.0, -5.0, 5.0, -4.0, 4.0)")
        bounds = (-5.0, 5.0, -5.0, 5.0, -4.0, 4.0)

    run = generate_circular_trajectory(num_drones, num_frames, duration, bounds, seed)
    output_path = (Path(output).with_suffix('.json') if Path(output).suffix.lower() != '.json'
                  else Path(output))
    save_simulation_run(run, output_path)
    click.echo(f"✓ Generated {len(run.frames)} frames → {output_path}")

