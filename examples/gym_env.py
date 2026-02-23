import sys
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium import spaces
from flockrl_sim.environment import EnvironmentSpecLoader, EnvironmentBuilder
from flockrl_sim.simulator import CoreSimulator
from flockrl_sim.state import SwarmState
from flockrl_sim.config import SimulationConfig
from flockrl_sim.collision.system import CollisionSystem
from stable_baselines3.common.env_util import make_vec_env

HAS_SB3 = True

class FlockRLEnv(gym.Env):
    """Single-file Gymnasium wrapper around CoreSimulator.
    Obs: flattened positions + velocities (N * 6)
    Action: flattened accelerations (N * 3)
    """

    metadata = {"render_modes": []}

    def __init__(self, spec="simple", num_drones: int = 1, max_acc: float = 2.0):
        loader = EnvironmentSpecLoader()
        # support both loader.load_preset (if present) or loader.load
        spec_obj = loader.load_preset(spec) if hasattr(loader, "load_preset") else loader.load(spec)
        env = EnvironmentBuilder.from_spec(spec_obj).build()

        sim_config = SimulationConfig(delta_t=1.0 / 30.0, max_acceleration=max_acc)
        collision = CollisionSystem(env)
        self.sim = CoreSimulator(delta_t=sim_config.delta_t, collision_system=collision, environment=env, config=sim_config)

        x_min, x_max, y_min, y_max, z_min, z_max = env.bounds

        # Spawn safely away from bounds and above low obstacles
        margin_xy = 1.0  # meters away from walls
        safe_x_min = x_min + margin_xy
        safe_x_max = x_max - margin_xy
        safe_y_min = y_min + margin_xy
        safe_y_max = y_max - margin_xy
        # Choose center positions if single drone; spread linearly if multiple
        if num_drones == 1:
            xs = np.array([(safe_x_min + safe_x_max) / 2])  # center: 0
            ys = np.array([(safe_y_min + safe_y_max) / 2])  # center: 0
        else:
            xs = np.linspace(safe_x_min, safe_x_max, num_drones)
            ys = np.linspace(safe_y_min, safe_y_max, num_drones)
        # Safe Z: at least 2m or z_min+1.5, clamped below ceiling
        safe_z = min(z_max - 0.5, max(z_min + 1.5, 2.0))
        zs = np.full(num_drones, safe_z)
        pos = np.column_stack([xs, ys, zs])

        ids = np.arange(num_drones)
        # Set goals away from start so there is something to do (opposite side)
        goal_xs = np.linspace(safe_x_max, safe_x_min, num_drones)
        goal_ys = np.linspace(safe_y_max, safe_y_min, num_drones)
        goal_zs = np.full(num_drones, safe_z)
        goals = np.column_stack([goal_xs, goal_ys, goal_zs])

        init_state = SwarmState.from_initial_positions(pos, ids, goals)
        
        self.num_drones = num_drones
        self._initial_state = init_state
        self.sim.start_run(initial_state=init_state, metadata={"spec": spec})

        self.action_space = spaces.Box(low=-max_acc, high=max_acc, shape=(num_drones * 3,), dtype=np.float32)
        obs_low = -np.inf * np.ones(num_drones * 6, dtype=np.float32)
        obs_high = np.inf * np.ones(num_drones * 6, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.start_run(initial_state=self._initial_state, metadata={})
        s = self.sim.state
        obs = np.concatenate([s.pos.flatten(), s.vel.flatten()]).astype(np.float32)
        return obs, {}

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(self.num_drones, 3)
        state, info = self.sim.step(a)
        obs = np.concatenate([state.pos.flatten(), state.vel.flatten()]).astype(np.float32)
        reward = -float(np.mean(np.linalg.norm(state.pos - state.goals, axis=1)))
        terminated = bool(info.get("done", False))
        truncated = False
        return obs, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass


def smoke_test():
    env = FlockRLEnv(spec="simple", num_drones=1)
    obs, _ = env.reset()
    print("Reset obs shape:", obs.shape)
    act = env.action_space.sample()
    obs2, rew, term, trunc, info = env.step(act)
    print("Step -> reward:", rew, "done:", term, "info keys:", list(info.keys()))


def train_with_sb3(total_timesteps: int = 20000):
    if not HAS_SB3:
        print("stable-baselines3 not available. Install with: pip install stable-baselines3 torch")
        return
    def make_env():
        return FlockRLEnv(spec="simple", num_drones=1)
    vec_env = make_vec_env(make_env, n_envs=1)
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save("ppo_flockrl_example")
    print("Training complete, model saved as ppo_flockrl_example.zip")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
        train_with_sb3(total_timesteps=steps)
    else:
        smoke_test()