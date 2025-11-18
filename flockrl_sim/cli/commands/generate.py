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

    x_span = x_max - x_min
    y_span = y_max - y_min
    z_span = z_max - z_min
    if x_span <= 0 or y_span <= 0:
        raise ValueError("Environment bounds must have positive XY span for circular trajectories")

    x_margin = min(3.0, x_span / 4)
    y_margin = min(3.0, y_span / 4)
    z_margin = min(1.0, z_span / 4)

    max_radius = min(3.0, x_margin, y_margin)
    if max_radius <= 0:
        raise ValueError("Environment bounds are too tight for circular trajectories")
    min_radius = 1.0 if max_radius >= 1.0 else max(0.1, max_radius * 0.5)
    if min_radius > max_radius:
        min_radius = max_radius * 0.5

    if np.isclose(max_radius, min_radius):
        radii = np.full(num_drones, max_radius)
    else:
        radii = np.random.uniform(min_radius, max_radius, num_drones)

    def sample_axis(low_vals, high_vals):
        lows = np.asarray(low_vals)
        highs = np.asarray(high_vals)
        centers = np.empty_like(lows)
        for idx, (low, high) in enumerate(zip(lows, highs)):
            centers[idx] = (low + high) / 2.0 if high <= low else np.random.uniform(low, high)
        return centers

    center_x = sample_axis(x_min + radii, x_max - radii)
    center_y = sample_axis(y_min + radii, y_max - radii)

    z_low = z_min + z_margin
    z_high = z_max - z_margin
    if z_high <= z_low:
        center_z = np.full(num_drones, (z_low + z_high) / 2.0)
    else:
        center_z = np.random.uniform(z_low, z_high, num_drones)

    angular_speeds = np.random.uniform(0.5, 2.0, num_drones)
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
    except Exception as exc:
        raise click.ClickException(f"Failed to load environment '{environment}': {exc}") from exc

    bounds = spec.bounds
    click.echo(f"Using environment: {spec.name}")
    click.echo(f"Bounds: {bounds}")

    run = generate_circular_trajectory(num_drones, num_frames, duration, bounds, seed)
    output_path = (Path(output).with_suffix('.json') if Path(output).suffix.lower() != '.json'
                  else Path(output))
    save_simulation_run(run, output_path)
    click.echo(f"✓ Generated {len(run.frames)} frames → {output_path}")
