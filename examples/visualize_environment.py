import sys
import matplotlib.pyplot as plt
from flockrl_sim.environment import EnvironmentSpecLoader, EnvironmentBuilder
from collections import Counter
from examples.visualization_utils import get_obstacle_props, draw_spawn_markers, draw_3d_box, draw_bounds, draw_top_face


def draw_topdown_view(ax, env, spec):
    x_min, x_max, y_min, y_max, *_ = env.bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min],
           'k--', linewidth=2, alpha=0.5, label='Bounds')

    for obs in env.obstacles:
        color, alpha, dims = get_obstacle_props(obs)
        draw_top_face(ax, obs, dims, color, alpha)

    draw_spawn_markers(ax, spec.spawn_zones, is_3d=False)
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Top-Down View', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)


def visualize_environment(spec_name_or_path):
    try:
        spec = EnvironmentSpecLoader().load(spec_name_or_path)
        print(f"Loaded: {spec.name}\nDescription: {spec.description}\nBounds: {spec.bounds}")
    except Exception as e:
        print(f"Error loading '{spec_name_or_path}': {e}")
        sys.exit(1)

    env = EnvironmentBuilder.from_spec(spec).build()
    counts = Counter(obs.type for obs in env.obstacles)
    print(f"Total obstacles: {len(env.obstacles)}")
    for t, c in counts.items():
        print(f"  - {t}: {c}")

    fig = plt.figure(figsize=(18, 8))
    ax3d = fig.add_subplot(121, projection='3d')
    x_min, x_max, y_min, y_max, z_min, z_max = env.bounds
    ax3d.set_xlim(x_min, x_max)
    ax3d.set_ylim(y_min, y_max)
    ax3d.set_zlim(z_min, z_max)
    draw_bounds(ax3d, env.bounds)

    for obs in env.obstacles:
        color, alpha, dims = get_obstacle_props(obs)
        draw_3d_box(ax3d, obs.position, dims, obs.orientation, color, alpha)

    draw_spawn_markers(ax3d, spec.spawn_zones, is_3d=True)
    ax3d.set_xlabel('X (m)', fontsize=12)
    ax3d.set_ylabel('Y (m)', fontsize=12)
    ax3d.set_zlabel('Z (m)', fontsize=12)
    ax3d.set_title(f'3D View\nEnvironment: {spec.name} ({len(env.obstacles)} obstacles)',
                  fontsize=12, fontweight='bold')
    ax3d.legend(loc='upper right', fontsize=9)
    ax3d.view_init(elev=20, azim=45)

    draw_topdown_view(fig.add_subplot(122), env, spec)
    plt.tight_layout()
    return fig, ax3d
