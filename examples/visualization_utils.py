import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from flockrl_sim.environment.obstacles_types import Wall, Gate, RectangularPrism

# Unified obstacle configuration with type-based styling
OBSTACLE_PROPS = {
    Wall: {'color': '#D3D3D3', 'alpha': 0.25, 'dims': lambda o: (o.length, o.thickness, o.height)},
    Gate: {'color': '#FFD700', 'alpha': 0.95, 'dims': lambda o: (o.width, o.thickness, o.height)},
    RectangularPrism: {'color': 'brown', 'alpha': 0.6, 'dims': lambda o: (o.length, o.width, o.height)},
}
DEFAULT_OBSTACLE = {'color': 'gray', 'alpha': 0.5, 'dims': lambda o: (1, 1, 1)}

# Unified spawn marker configuration
SPAWN_MARKERS = {
    'start_position': {'color': 'blue', 'marker': 'o', 'label': 'Start', 'size_2d': 200, 'size_3d': 100},
    'goal_position': {'color': 'red', 'marker': '*', 'label': 'Goal', 'size_2d': 300, 'size_3d': 100},
}


def get_obstacle_props(obs):
    """Returns color, alpha, and dimensions for an obstacle."""
    props = OBSTACLE_PROPS.get(type(obs), DEFAULT_OBSTACLE)
    return props['color'], props['alpha'], props['dims'](obs)


def create_box_vertices(center, dimensions, orientation=None):
    """Create 3D box vertices with optional rotation."""
    x, y, z = center
    l, w, h = dimensions

    # Define box vertices centered at origin
    vertices = np.array([
        [-l/2, -w/2, -h/2], [+l/2, -w/2, -h/2],
        [+l/2, +w/2, -h/2], [-l/2, +w/2, -h/2],
        [-l/2, -w/2, +h/2], [+l/2, -w/2, +h/2],
        [+l/2, +w/2, +h/2], [-l/2, +w/2, +h/2]
    ])

    # Apply yaw rotation if orientation provided
    if orientation and len(orientation) >= 3 and orientation[2] != 0.0:
        yaw = orientation[2]
        rotation = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])
        vertices = vertices @ rotation.T

    # Translate to center position
    return vertices + np.array([x, y, z])


def create_box_faces(vertices):
    """Create face indices for a box from vertices."""
    face_indices = [(0,1,5,4), (2,3,7,6), (0,3,7,4), (1,2,6,5), (0,1,2,3), (4,5,6,7)]
    return [[vertices[i] for i in idx] for idx in face_indices]


def draw_spawn_markers(ax, env, is_3d=False):
    """Draw start/goal markers on 2D or 3D axes using resolved environment positions."""
    if env:
        for attr, config in SPAWN_MARKERS.items():
            if position := getattr(env, attr, None):
                coords = position if is_3d else position[:2]
                size = config['size_3d' if is_3d else 'size_2d']
                ax.scatter(
                    *coords,
                    color=config['color'],
                    s=size,
                    marker=config['marker'],
                    label=config['label'],
                    edgecolors='black' if not is_3d else None,
                    linewidths=2 if not is_3d else None,
                    zorder=10
                )


def draw_top_face(ax, obs, dims, color, alpha, **kwargs):
    """Draw the obstacle's top face projected onto XY plane."""
    vertices = create_box_vertices(obs.position, dims, obs.orientation)
    coords = vertices[[0, 1, 2, 3], :2]
    ax.fill(
        coords[:, 0],
        coords[:, 1],
        facecolor=color,
        alpha=alpha,
        edgecolor=kwargs.get('edgecolor', 'black'),
        linewidth=kwargs.get('linewidth', 1),
    )


def draw_3d_box(ax, position, dimensions, orientation=None, color='gray', alpha=0.5):
    """Draw a 3D box (rectangular prism) on a 3D axis."""
    vertices = create_box_vertices(position, dimensions, orientation)
    faces = create_box_faces(vertices)
    ax.add_collection3d(Poly3DCollection(
        faces,
        alpha=alpha,
        facecolor=color,
        edgecolor='black',
        linewidth=0.5
    ))


def draw_bounds(ax, bounds, color='black', linestyle='--', alpha=0.5):
    """Draw bounding box wireframe on a 3D axis."""
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    kw = dict(color=color, linestyle=linestyle, alpha=alpha, linewidth=2)

    # Draw horizontal edges at z_min and z_max
    for z in [z_min, z_max]:
        for xs, ys in [([x_min, x_max], [y_min, y_min]),
                       ([x_min, x_max], [y_max, y_max]),
                       ([x_min, x_min], [y_min, y_max]),
                       ([x_max, x_max], [y_min, y_max])]:
            ax.plot(xs, ys, [z] * len(xs), **kw)

    # Draw vertical edges
    for x, y in [(x_min, y_min), (x_max, y_min), (x_min, y_max), (x_max, y_max)]:
        ax.plot([x, x], [y, y], [z_min, z_max], **kw)
