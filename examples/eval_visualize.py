import sys
import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from gym_env import FlockRLEnv
HAS_SB3 = True

try:
    from visualization_utils import (
        get_obstacle_props,
        draw_top_face,
        draw_3d_box,
        draw_spawn_markers,
    )
    HAS_VIZ_UTILS = True
except Exception:
    HAS_VIZ_UTILS = False


def _rollout(env: FlockRLEnv, model, max_steps: int, deterministic: bool = True):
    """Run one episode; return per-agent trajectory array shape (N_agents, T, 3) and last info."""
    obs, _ = env.reset()
    traj = [[] for _ in range(env.num_drones)]
    last_info = {}
    for t in range(max_steps):
        if model is not None:
            action, _ = model.predict(obs, deterministic=deterministic)
        else:
            action = np.random.uniform(-2.0, 2.0, size=env.action_space.shape)  # <-- change this line
        obs, reward, terminated, truncated, info = env.step(action)
        state = env.sim.state
        for i in range(env.num_drones):
            traj[i].append(state.pos[i].copy())
        last_info = info
        if terminated or truncated:
            break
    return np.array(traj, dtype=float), last_info


def run_eval(model_path="ppo_flockrl_example", spec="simple", n_episodes=3, max_steps=500, mode="2d"):
    # Load model if available
    model = None
    if HAS_SB3:
        try:
            # accept either "name" or "name.zip"
            if os.path.exists(model_path) or os.path.exists(model_path + ".zip"):
                model = PPO.load(model_path)
            else:
                print(f"[warn] Model '{model_path}' not found, using random actions.")
        except Exception as e:
            print(f"[warn] Could not load model: {e}. Using random actions.")
            model = None
    else:
        print("[warn] stable-baselines3 not installed, using random actions.")

    env = FlockRLEnv(spec=spec, num_drones=1)

    # Roll out episodes and collect trajectories
    episodes = []
    for ep in range(n_episodes):
        traj, info = _rollout(env, model, max_steps)
        if traj.shape[1] <= 1:
            print(f"Episode {ep}: short trajectory ({traj.shape[1]} step). reason={info.get('termination_reason')}, collisions={info.get('collisions')}")
        else:
            print(f"Episode {ep}: T={traj.shape[1]} steps. reason={info.get('termination_reason')}, collisions={info.get('collisions')}")
        episodes.append((traj, info))

    # Plot
    if mode.lower() in ("3d", "3"):
        _plot_3d(episodes, env)
    else:
        _plot_2d(episodes, env)


def _plot_2d(episodes, env: FlockRLEnv):
    plt.figure(figsize=(7, 7))
    ax = plt.gca()
    x_min, x_max, y_min, y_max, *_ = env.sim.environment.bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    # Obstacles (top faces)
    if HAS_VIZ_UTILS:
        for obs in env.sim.environment.obstacles:
            color, alpha, dims = get_obstacle_props(obs)
            draw_top_face(ax, obs, dims, color, alpha)

    # Plot all episode trajectories
    for ep_idx, (traj, _) in enumerate(episodes):
        for agent_idx in range(traj.shape[0]):
            a = traj[agent_idx]
            if a.shape[0] == 0:
                continue
            ax.plot(a[:, 0], a[:, 1], "-", lw=2, alpha=0.9, label=f"ep{ep_idx}_agent{agent_idx}")
            ax.scatter(a[0, 0], a[0, 1], c="green", marker="s", s=60 if ep_idx == 0 else 40)

    # Start/goal markers
    goals = env._initial_state.goals
    for i in range(env.num_drones):
        ax.scatter(goals[i, 0], goals[i, 1], c="red", marker="*", s=140, label="goal" if i == 0 else None)

    ax.set_title("Top-down agent trajectories")
    ax.legend(loc="upper right", fontsize=8)
    plt.show()


def _plot_3d(episodes, env: FlockRLEnv):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # ensures 3D projection is registered

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    x_min, x_max, y_min, y_max, z_min, z_max = env.sim.environment.bounds
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    # Obstacles as boxes
    if HAS_VIZ_UTILS:
        for obs in env.sim.environment.obstacles:
            color, alpha, dims = get_obstacle_props(obs)
            draw_3d_box(ax, obs, dims, color, alpha)

    # Trajectories
    for ep_idx, (traj, _) in enumerate(episodes):
        for agent_idx in range(traj.shape[0]):
            a = traj[agent_idx]
            if a.shape[0] == 0:
                continue
            ax.plot(a[:, 0], a[:, 1], a[:, 2], lw=2, alpha=0.9, label=f"ep{ep_idx}_agent{agent_idx}")
            ax.scatter(a[0, 0], a[0, 1], a[0, 2], c="green", marker="s", s=60 if ep_idx == 0 else 40)

    # Goal markers
    goals = env._initial_state.goals
    for i in range(env.num_drones):
        ax.scatter(goals[i, 0], goals[i, 1], goals[i, 2] if goals.shape[1] == 3 else (z_min + z_max) / 2.0,
                   c="red", marker="*", s=140, label="goal" if i == 0 else None)

    ax.set_title("3D agent trajectories")
    ax.legend(loc="upper right", fontsize=8)
    plt.show()


if __name__ == "__main__":
    # Usage:
    #   python examples/eval_visualize.py            # 2D, default model name
    #   python examples/eval_visualize.py 3d         # 3D
    #   python examples/eval_visualize.py 2d mymodel 5 400
    mode = "2d"
    model_path = "ppo_flockrl_example"
    n_episodes = 3
    max_steps = 500

    if len(sys.argv) >= 2:
        arg = sys.argv[1].lower()
        if arg in ("2d", "3d", "2", "3"):
            mode = "3d" if arg in ("3d", "3") else "2d"
        else:
            model_path = sys.argv[1]
    if len(sys.argv) >= 3:
        # Could be model path or mode; handle both
        if sys.argv[2].lower() in ("2d", "3d", "2", "3"):
            mode = "3d" if sys.argv[2].lower() in ("3d", "3") else "2d"
        else:
            model_path = sys.argv[2]
    if len(sys.argv) >= 4:
        n_episodes = int(sys.argv[3])
    if len(sys.argv) >= 5:
        max_steps = int(sys.argv[4])

    run_eval(model_path=model_path, spec="simple", n_episodes=n_episodes, max_steps=max_steps, mode=mode)