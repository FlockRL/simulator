import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from flockrl_sim.environment import EnvironmentSpecLoader, EnvironmentBuilder


def draw_rectangular_prism(ax, position, length, width, height, color='gray', alpha=0.3):
    x, y, z = position
    vertices = np.array([
        [x-length/2, y-width/2, z], [x+length/2, y-width/2, z],
        [x+length/2, y+width/2, z], [x-length/2, y+width/2, z],
        [x-length/2, y-width/2, z+height], [x+length/2, y-width/2, z+height],
        [x+length/2, y+width/2, z+height], [x-length/2, y+width/2, z+height]
    ])
    faces = [[vertices[i] for i in idx] for idx in
            [(0,1,5,4), (2,3,7,6), (0,3,7,4), (1,2,6,5), (0,1,2,3), (4,5,6,7)]]
    ax.add_collection3d(Poly3DCollection(faces, alpha=alpha, facecolor=color,
                                        edgecolor='black', linewidth=0.5))

def draw_wall(ax, wall, color='brown', alpha=0.5):
    draw_rectangular_prism(ax, wall.position, wall.length, wall.thickness, wall.height, color, alpha)

def draw_gate(ax, gate, color='lightblue', alpha=0.3):
    draw_rectangular_prism(ax, gate.position, gate.width, gate.thickness, gate.height, color, alpha)

def draw_clutter(ax, clutter, color='gray', alpha=0.6):
    if hasattr(clutter, 'length'):
        draw_rectangular_prism(ax, clutter.position, clutter.length, clutter.width, clutter.height, color, alpha)


def draw_bounds(ax, bounds, color='black', linestyle='--', alpha=0.5):
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    kw = dict(color=color, linestyle=linestyle, alpha=alpha, linewidth=2)

    for z in [z_min, z_max]:
        for xs, ys in [([x_min,x_max], [y_min,y_min]), ([x_min,x_max], [y_max,y_max]),
                       ([x_min,x_min], [y_min,y_max]), ([x_max,x_max], [y_min,y_max])]:
            ax.plot(xs, ys, [z]*len(xs), **kw)

    for x, y in [(x_min,y_min), (x_max,y_min), (x_min,y_max), (x_max,y_max)]:
        ax.plot([x,x], [y,y], [z_min,z_max], **kw)


def draw_topdown_view(ax, env, spec):
    x_min, x_max, y_min, y_max, *_ = env.bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min],
           'k--', linewidth=2, alpha=0.5, label='Bounds')

    for obs in env.obstacles:
        x, y = obs.position[:2]
        params = {
            'wall': (x-obs.length/2, y-obs.thickness/2, obs.length, obs.thickness,
                    'brown', 'black', 0.6),
            'gate': (x-obs.width/2, y-obs.thickness/2, obs.width, obs.thickness,
                    'lightblue', 'blue', 0.4),
            'clutter': (x-obs.length/2, y-obs.width/2, obs.length, obs.width,
                       'gray', 'black', 0.5) if hasattr(obs, 'length') else None
        }
        if obs.type in params and params[obs.type]:
            px, py, w, h, fc, ec, a = params[obs.type]
            ax.add_patch(plt.Rectangle((px, py), w, h, facecolor=fc, edgecolor=ec,
                                      alpha=a, linewidth=1))

    if spec.spawn_zones:
        if spec.spawn_zones.start_position:
            ax.scatter(*spec.spawn_zones.start_position[:2], color='green', s=200, marker='o',
                      edgecolors='black', linewidths=2, label='Start', zorder=10)
        if spec.spawn_zones.goal_position:
            ax.scatter(*spec.spawn_zones.goal_position[:2], color='red', s=200, marker='*',
                      edgecolors='black', linewidths=2, label='Goal', zorder=10)

    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_title('Top-Down View', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)


def visualize_environment(spec_name_or_path):
    loader = EnvironmentSpecLoader()
    try:
        spec = loader.load(spec_name_or_path)
        print(f"Loaded: {spec.name}")
        print(f"Description: {spec.description}")
        print(f"Bounds: {spec.bounds}")
    except Exception as e:
        print(f"Error loading '{spec_name_or_path}': {e}")
        sys.exit(1)

    env = EnvironmentBuilder.from_spec(spec).build()
    counts = {}
    for obs in env.obstacles:
        counts[obs.type] = counts.get(obs.type, 0) + 1

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
        draw_func = {'wall': draw_wall, 'gate': draw_gate, 'clutter': draw_clutter}.get(obs.type)
        if draw_func:
            draw_func(ax3d, obs)

    if spec.spawn_zones:
        if spec.spawn_zones.start_position:
            ax3d.scatter(*spec.spawn_zones.start_position, color='green', s=200, marker='o',
                        edgecolors='black', linewidths=2, label='Start')
        if spec.spawn_zones.goal_position:
            ax3d.scatter(*spec.spawn_zones.goal_position, color='red', s=200, marker='*',
                        edgecolors='black', linewidths=2, label='Goal')

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

