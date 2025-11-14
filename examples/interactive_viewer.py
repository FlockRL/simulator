#!/usr/bin/env python3
import sys
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

try:
    from simple_term_menu import TerminalMenu
    HAS_TERM_MENU = True
except ImportError:
    HAS_TERM_MENU = False

from flockrl_sim.environment import EnvironmentSpecLoader, EnvironmentBuilder
from flockrl_sim.environment.obstacles_types import Wall, Gate, RectangularPrism

def select_view_mode():
    options = ["2D + 3D (Unified)", "2D Only", "3D Only"]

    if HAS_TERM_MENU:
        menu = TerminalMenu(options, menu_cursor="➤ ")
        idx = menu.show()
        return ['unified', '2d', '3d'][idx if idx is not None else 0]

    # Fallback to simple numbered menu if simple-term-menu is not available
    print("Select view mode:")
    for i, opt in enumerate(options, 1):
        print(f"{i}. {opt}")

    while True:
        try:
            choice = int(input("Enter choice (1-3): ").strip())
            if 1 <= choice <= 3:
                return ['unified', '2d', '3d'][choice - 1]
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except (ValueError, KeyboardInterrupt):
            print("\nDefaulting to unified view...")
            return 'unified'


def create_box_vertices(center, dimensions, orientation):
    x, y, z, l, w, h = *center, *dimensions
    vertices = np.array([[-l/2, -w/2, -h/2], [+l/2, -w/2, -h/2], [+l/2, +w/2, -h/2], [-l/2, +w/2, -h/2],
                         [-l/2, -w/2, +h/2], [+l/2, -w/2, +h/2], [+l/2, +w/2, +h/2], [-l/2, +w/2, +h/2]])
    if orientation and len(orientation) >= 3:
        yaw = orientation[2]
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        vertices = vertices @ rotation.T
    return vertices + np.array([x, y, z])


def get_obstacle_color_alpha(obs):
    if isinstance(obs, Wall):
        return ('#D3D3D3', 0.25)
    if isinstance(obs, Gate):
        return ('#FFD700', 0.95)
    if isinstance(obs, RectangularPrism):
        return ('brown', 0.6)
    return ('gray', 0.5)

def get_obstacle_dimensions(obs):
    if isinstance(obs, Wall):
        return (obs.length, obs.thickness, obs.height)
    if isinstance(obs, Gate):
        return (obs.width, obs.thickness, obs.height)
    if isinstance(obs, RectangularPrism):
        return (obs.length, obs.width, obs.height)
    return (1, 1, 1)


def draw_spawn_markers(ax, spawn_zones, is_3d=False):
    if not spawn_zones:
        return
    if spawn_zones.start_position:
        coords = spawn_zones.start_position if is_3d else spawn_zones.start_position[:2]
        ax.scatter(*coords, color='blue', s=100 if is_3d else 200, marker='o', label='Start', zorder=10)
    if spawn_zones.goal_position:
        coords = spawn_zones.goal_position if is_3d else spawn_zones.goal_position[:2]
        ax.scatter(*coords, color='red', s=100 if is_3d else 300, marker='*', label='Goal', zorder=10)


def draw_2d_slice(ax, env, spec, z_level):
    ax.clear()
    x_min, x_max, y_min, y_max, *_ = env.bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')

    for obs in env.obstacles:
        dims = get_obstacle_dimensions(obs)
        obs_z_min, obs_z_max = obs.position[2] - dims[2]/2, obs.position[2] + dims[2]/2
        if obs_z_min <= z_level <= obs_z_max:
            vertices = create_box_vertices(obs.position, dims, obs.orientation)
            color, alpha = get_obstacle_color_alpha(obs)
            ax.fill(vertices[[0,1,2,3], 0], vertices[[0,1,2,3], 1],
                   color=color, alpha=alpha, edgecolor='black', linewidth=1)

    draw_spawn_markers(ax, getattr(spec, "spawn_zones", None))
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
        vertices = create_box_vertices(obs.position, get_obstacle_dimensions(obs), obs.orientation)
        faces = [[vertices[i] for i in idx] for idx in
                [(0,1,5,4), (2,3,7,6), (0,3,7,4), (1,2,6,5), (0,1,2,3), (4,5,6,7)]]
        color, alpha = get_obstacle_color_alpha(obs)
        ax.add_collection3d(Poly3DCollection(faces, alpha=alpha, facecolor=color,
                                            edgecolor='black', linewidth=0.5))

    draw_spawn_markers(ax, getattr(spec, "spawn_zones", None), is_3d=True)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'{spec.name} - 3D View', fontweight='bold')
    ax.legend()


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
        self.load_environment((self.current_idx + direction) % len(self.presets))


class Viewer2D(BaseViewer):
    def __init__(self):
        super().__init__()
        self.z_level, self.z_step = 0.0, 0.5
        self.fig = plt.figure(figsize=(12, 10))
        self.fig.canvas.manager.set_window_title('FlockRL 2D Viewer')
        self.ax = self.fig.add_subplot(111)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

        print("\n" + "=" * 60)
        print("2D VIEWER CONTROLS")
        print("=" * 60)
        print("  ← → : Cycle through environments")
        print("  ↑ ↓ : Move slice up/down (z-axis)")
        print("  Q   : Quit")
        print("=" * 60 + "\n")

        self.render()
        plt.tight_layout()
        plt.show()

    def on_key(self, event):
        if event.key in ('left', 'right'):
            self.cycle_environment(-1 if event.key == 'left' else 1)
            self.render()
        elif event.key in ('up', 'down'):
            self.z_level = (min(self.z_level + self.z_step, self.z_max) if event.key == 'up'
                          else max(self.z_level - self.z_step, self.z_min))
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

        print("\n" + "=" * 60)
        print("3D VIEWER CONTROLS")
        print("=" * 60)
        print("  ← →   : Cycle through environments")
        print("  Mouse : Drag=rotate")
        print("  Q     : Quit")
        print("=" * 60 + "\n")

        self.render()
        plt.tight_layout()
        plt.show()

    def on_key(self, event):
        if event.key in ('left', 'right'):
            elev, azim = self.ax.elev, self.ax.azim
            self.cycle_environment(-1 if event.key == 'left' else 1)
            self.render()
            self.ax.view_init(elev=elev, azim=azim)
            self.fig.canvas.draw_idle()
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

        print("\n" + "=" * 60)
        print("UNIFIED VIEWER CONTROLS")
        print("=" * 60)
        print("  ← → : Cycle through environments")
        print("  ↑ ↓ : Move 2D slice up/down (z-axis)")
        print("  Mouse (3D panel): Left-drag=rotate, Right-drag=zoom")
        print("  Q   : Quit")
        print("=" * 60 + "\n")

        self.render()
        plt.tight_layout()
        plt.show()

    def on_key(self, event):
        if event.key in ('left', 'right'):
            self.cycle_environment(-1 if event.key == 'left' else 1)
            self.render()
        elif event.key in ('up', 'down'):
            self.z_level = (min(self.z_level + self.z_step, self.z_max) if event.key == 'up'
                          else max(self.z_level - self.z_step, self.z_min))
            self.render_2d()
            self.fig.canvas.draw_idle()
        elif event.key == 'q':
            plt.close(self.fig)

    def render(self):
        elev, azim = (self.ax_3d.elev, self.ax_3d.azim) if hasattr(self.ax_3d, 'elev') else (20, -60)
        self.render_2d()
        self.render_3d()
        self.ax_3d.view_init(elev=elev, azim=azim)
        self.fig.canvas.draw_idle()

    def render_2d(self):
        draw_2d_slice(self.ax_2d, self.env, self.spec, self.z_level)

    def render_3d(self):
        draw_3d_scene(self.ax_3d, self.env, self.spec)


def main():
    print("Select a view mode:")
    mode = select_view_mode()
    {'unified': ViewerUnified, '2d': Viewer2D, '3d': Viewer3D}[mode]()

if __name__ == "__main__":
    main()
