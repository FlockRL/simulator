#!/usr/bin/env python3
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from flockrl_sim.environment import EnvironmentSpecLoader, EnvironmentBuilder
from visualization_utils import get_obstacle_props, draw_spawn_markers, draw_3d_box, draw_top_face
try:
    from simple_term_menu import TerminalMenu
    HAS_TERM_MENU = True
except ImportError:
    HAS_TERM_MENU = False

VIEW_MODES = ['unified', '2d', '3d']
VIEW_OPTIONS = ["2D + 3D (Unified)", "2D Only", "3D Only"]

VIEWER_CONTROLS = {
    '2d': ["2D VIEWER CONTROLS", "  ← → : Cycle through environments", "  ↑ ↓ : Move slice up/down (z-axis)", "  Q   : Quit"],
    '3d': ["3D VIEWER CONTROLS", "  ← →   : Cycle through environments", "  Mouse : Drag=rotate", "  Q     : Quit"],
    'unified': ["UNIFIED VIEWER CONTROLS", "  ← → : Cycle through environments", "  ↑ ↓ : Move 2D slice up/down (z-axis)", "  Mouse (3D panel): Left-drag=rotate, Right-drag=zoom", "  Q   : Quit"]
}


def select_view_mode():
    if HAS_TERM_MENU:
        idx = TerminalMenu(VIEW_OPTIONS, menu_cursor="➤ ").show()
        return VIEW_MODES[idx if idx is not None else 0]

    print("Select view mode:")
    for i, opt in enumerate(VIEW_OPTIONS, 1):
        print(f"{i}. {opt}")

    try:
        choice = int(input("Enter choice (1-3): ").strip())
        if 1 <= choice <= 3:
            return VIEW_MODES[choice - 1]
        print("Invalid choice. Please enter 1, 2, or 3.")
        return select_view_mode()
    except (ValueError, KeyboardInterrupt):
        print("\nDefaulting to unified view...")
        return 'unified'


def draw_2d_slice(ax, env, spec, z_level):
    ax.clear()
    x_min, x_max, y_min, y_max, *_ = env.bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')

    for obs in env.obstacles:
        color, alpha, dims = get_obstacle_props(obs)
        obs_z_min, obs_z_max = obs.position[2] - dims[2]/2, obs.position[2] + dims[2]/2
        if obs_z_min <= z_level <= obs_z_max:
            draw_top_face(ax, obs, dims, color, alpha)

    draw_spawn_markers(ax, env)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f'{spec.name} - 2D Slice at Z = {z_level:.1f}m', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()


def draw_3d_scene(ax, env, spec):
    ax.clear()
    x_min, x_max, y_min, y_max, z_min, z_max = env.bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)

    for obs in env.obstacles:
        color, alpha, dims = get_obstacle_props(obs)
        draw_3d_box(ax, obs.position, dims, obs.orientation, color, alpha)

    draw_spawn_markers(ax, env, is_3d=True)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'{spec.name} - 3D View', fontweight='bold')
    ax.legend()


def print_controls(mode):
    """Print viewer control instructions."""
    lines = VIEWER_CONTROLS[mode]
    sep = "=" * 60
    print(f"\n{sep}\n{lines[0]}\n{sep}")
    print('\n'.join(lines[1:]))
    print(f"{sep}\n")

class BaseViewer:
    def __init__(self):
        self.loader = EnvironmentSpecLoader()
        self.presets = self.loader.list_presets()
        if not self.presets:
            raise RuntimeError("No environment presets available")
        self.current_idx = 0
        self.load_environment(0)

    def load_environment(self, idx):
        self.current_idx = idx % len(self.presets)
        preset_name = self.presets[self.current_idx]
        self.spec = self.loader.load_preset(preset_name)
        self.env = EnvironmentBuilder.from_spec(self.spec).build()
        self.bounds = self.env.bounds
        self.z_min, self.z_max = self.env.bounds[4], self.env.bounds[5]

    def cycle_environment(self, direction):
        self.load_environment(self.current_idx + direction)

    def handle_arrow_keys(self, key):
        """Common handler for left/right arrow keys. Returns True if handled."""
        if key in ('left', 'right'):
            self.cycle_environment(-1 if key == 'left' else 1)
        return key in ('left', 'right')


class Viewer2D(BaseViewer):
    def __init__(self):
        super().__init__()
        self.z_level, self.z_step = 0.0, 0.5
        self.fig = plt.figure(figsize=(12, 10))
        self.fig.canvas.manager.set_window_title('FlockRL 2D Viewer')
        self.ax = self.fig.add_subplot(111)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        print_controls('2d')
        self.render()
        plt.tight_layout()
        plt.show()

    def on_key(self, event):
        if self.handle_arrow_keys(event.key):
            self.render()
        elif event.key in ('up', 'down'):
            delta = self.z_step if event.key == 'up' else -self.z_step
            self.z_level = np.clip(self.z_level + delta, self.z_min, self.z_max)
            self.render()
        elif event.key == 'q':
            plt.close(self.fig)

    def render(self):
        draw_2d_slice(self.ax, self.env, self.spec, self.z_level)
        self.fig.canvas.draw_idle()


class Viewer3D(BaseViewer):
    def __init__(self):
        super().__init__()
        self.fig = plt.figure(figsize=(12, 10))
        self.fig.canvas.manager.set_window_title('FlockRL 3D Viewer')
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        print_controls('3d')
        self.render()
        plt.tight_layout()
        plt.show()

    def on_key(self, event):
        if self.handle_arrow_keys(event.key):
            self.render()
            self.ax.view_init(elev=self.ax.elev, azim=self.ax.azim)
        elif event.key == 'q':
            plt.close(self.fig)

    def render(self):
        draw_3d_scene(self.ax, self.env, self.spec)
        self.fig.canvas.draw_idle()


class ViewerUnified(BaseViewer):
    def __init__(self):
        super().__init__()
        self.z_level, self.z_step = 0.0, 0.5
        self.fig = plt.figure(figsize=(18, 9))
        self.fig.canvas.manager.set_window_title('FlockRL Unified Viewer')
        self.ax_2d = self.fig.add_subplot(121)
        self.ax_3d = self.fig.add_subplot(122, projection='3d')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        print_controls('unified')
        self.render()
        plt.tight_layout()
        plt.show()

    def on_key(self, event):
        if self.handle_arrow_keys(event.key):
            self.render()
        elif event.key in ('up', 'down'):
            delta = self.z_step if event.key == 'up' else -self.z_step
            self.z_level = np.clip(self.z_level + delta, self.z_min, self.z_max)
            self.render_2d()
            self.fig.canvas.draw_idle()
        elif event.key == 'q':
            plt.close(self.fig)

    def render(self):
        elev, azim = (self.ax_3d.elev, self.ax_3d.azim) if hasattr(self.ax_3d, 'elev') else (20, -60)
        self.render_2d()
        self.render_3d()
        self.ax_3d.view_init(elev, azim)
        self.fig.canvas.draw_idle()

    def render_2d(self):
        draw_2d_slice(self.ax_2d, self.env, self.spec, self.z_level)

    def render_3d(self):
        draw_3d_scene(self.ax_3d, self.env, self.spec)


VIEWER_CLASSES = {'unified': ViewerUnified, '2d': Viewer2D, '3d': Viewer3D}

def main():
    print("Select a view mode:")
    mode = select_view_mode()
    VIEWER_CLASSES[mode]()

if __name__ == "__main__":
    main()
