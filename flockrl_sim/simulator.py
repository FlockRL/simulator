"""
Core simulator interface skeleton.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import warnings

import numpy as np

from .state import SwarmState
from .environment import Environment
from .config import SimulationConfig

import json
from .perception.sensors import PerceptionSystem

CollisionHandler = Callable[[SwarmState], Tuple[SwarmState, dict]]
RenderHook = Callable[[SwarmState, dict], None]


@dataclass
class SimulationFrame:
    """
    One snapshot of the swarm to be serialized for offline playback.
    """

    state: SwarmState
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationRun:
    """Collection of frames associated with a single simulator execution"""

    frames: List[SimulationFrame] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CoreSimulator:
    """Main simulator class that will handle the simulation loop (update state, apply collisions, etc.)"""

    def __init__(
        self,
        delta_t: float = 1.0 / 60.0,  # Defaults to 60 Hz
        collision_system: Optional[CollisionHandler] = None,
        render_hook: Optional[RenderHook] = None,
        environment: Optional[Environment] = None,
        config: Optional[SimulationConfig] = None,
        enable_logging: bool = True,
        enable_perception: bool = True,
    ) -> None:
        self.delta_t = delta_t
        self.collision_system = collision_system
        self.render_hook = render_hook
        # Default empty environment
        if environment is None:
            environment = Environment(
                bounds=(-100, 100, -100, 100, 0, 100),
                obstacles=[],
                start_position=(0.0, 0.0, 1.0),
                goal_position=(0.0, 0.0, 10.0),
                seed=0,
            )
        self.environment = environment
        self.config = config or SimulationConfig(delta_t=delta_t)
        self.state: Optional[SwarmState] = None  # Set by start_run()
        self.current_run: Optional[SimulationRun] = None

        # Episode management
        self._step_count = 0
        self._episode_terminated = False
        self._termination_reason: Optional[str] = None
        self._initial_state: Optional[SwarmState] = None

        # Episode statistics
        self._episode_stats = {
            "collision_count": 0,
            "min_goal_distance": float("inf"),
            "final_goal_distance": float("inf"),
            "total_steps": 0,
        }

        # Logging Toggle:
        self.enable_logging = enable_logging

        # Perception system - enabled by default for RL
        self._perception_system = None
        if self.environment is not None and enable_perception:
            self._perception_system = PerceptionSystem(
                environment=self.environment,
                config=None,  # Use default config
                seed=None,
            )

    def start_run(
        self,
        initial_state: Optional[SwarmState] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SwarmState:
        """
        Initialize a new simulation run with optional metadata.

        Replaces the internal swarm state and creates a new SimulationRun for logging.
        Callers are expected to invoke this before starting a new run that will
        later be saved for offline visualization.

        Args:
            initial_state: Optional initial state for the swarm. If None, uses default.
            metadata: Optional metadata to attach to the run (e.g., config, timestamp).

        Returns:
            The initialized SwarmState
        """
        # Determine goal position
        goal_pos = np.array([0.0, 0.0, 10.0], dtype=float)  # Default goal
        if (
            hasattr(self.environment, "goal_position")
            and self.environment.goal_position is not None
        ):
            goal_pos = np.array(self.environment.goal_position, dtype=float)

        if initial_state is None:
            # Create default state with single drone at origin
            positions = np.array([[0.0, 0.0, 1.0]])
            ids = np.array([0])
            goals = np.array([goal_pos])  # Single goal for single drone
            self.state = SwarmState.from_initial_positions(positions, ids, goals)
        else:
            self.state = initial_state.clone()
            # Ensure goals are set if not already present
            if self.state.goals is None:
                N = self.state.pos.shape[0] if self.state.pos is not None else 0
                if N > 0:
                    self.state.goals = np.tile(goal_pos, (N, 1))

        # Store initial state for reset
        self._initial_state = self.state.clone()

        # Reset episode management
        self._step_count = 0
        self._episode_terminated = False
        self._termination_reason = None

        # Reset episode statistics
        self._episode_stats = {
            "collision_count": 0,
            "min_goal_distance": float("inf"),
            "final_goal_distance": float("inf"),
            "total_steps": 0,
        }

        # Creating a run:
        self.current_run = SimulationRun(frames=[], metadata=metadata or {})

        # Get initial observations
        initial_observations = []
        if self._perception_system is not None:
            initial_observations = self._perception_system.observe(self.state)

        # Logging the first frame with consistent info structure
        if self.enable_logging:
            self.log_frame(
                info={
                    "event": "run_started",
                    "collisions": [],
                    "observations": initial_observations,
                    "step": 0,
                    "done": False,
                    "termination_reason": None,
                    "episode_stats": self._episode_stats.copy(),
                }
            )

        return self.state

    def step(self, actions: np.ndarray) -> tuple[SwarmState, dict]:
        """
        Advance the simulation by one tick and return the new state and info dict.

        Returns:
            Tuple of (updated SwarmState, info dict with collision events, termination status, etc.)
        """
        if self.state is None or self.state.pos is None:
            raise RuntimeError(
                "Simulator state is not initialized. Call start_run() first."
            )

        # Check if episode is already terminated
        if self._episode_terminated:
            raise RuntimeError(
                f"Episode already terminated: {self._termination_reason}. Call reset() to start a new episode."
            )

        # 1. Validate actions
        actions = self._validate_actions(actions)

        # 2. Apply actions as accelerations
        self.state.acc = actions

        # Calculate proposed new velocities and positions
        proposed_vel = self.state.vel + self.state.acc * self.delta_t
        proposed_pos = (
            self.state.pos
            + self.state.vel * self.delta_t
            + 0.5 * self.state.acc * (self.delta_t**2)
        )
        proposed_t = self.state.t + self.delta_t

        # 3. Create proposed state for collision checking
        proposed_state = self.state.clone()
        proposed_state.t = proposed_t
        proposed_state.pos = proposed_pos
        proposed_state.vel = proposed_vel
        proposed_state.acc = actions

        info_dict = {}
        final_state = proposed_state

        # 4. Call Collision System and apply collision response
        if self.collision_system:
            # The collision system detects collisions and returns collision info
            # We need to apply the new_position and rebound_velocity from CollisionInfo
            final_state, info_dict = self.collision_system(proposed_state)

            # Apply collision responses
            collisions = info_dict.get("collisions", [])
            if collisions:
                # Apply each collision's new_position and rebound_velocity
                for collision in collisions:
                    drone_id = collision.drone_id
                    # Find the index of this drone
                    drone_idx = np.where(final_state.ids == drone_id)[0]
                    if len(drone_idx) > 0:
                        idx = drone_idx[0]
                        final_state.pos[idx] = collision.new_position
                        final_state.vel[idx] = collision.rebound_velocity

                # Update statistics
                self._episode_stats["collision_count"] += len(collisions)

                # Terminate episode if configured to do so
                if self.config.terminate_on_collision:
                    self._episode_terminated = True
                    self._termination_reason = "collision"

        # 5. Update master state
        self.state = final_state

        # 6. Update step counter
        self._step_count += 1
        self._episode_stats["total_steps"] = self._step_count

        # 7. Update goal distance statistics
        goal_distances = np.linalg.norm(self.state.pos - self.state.goals, axis=1)
        min_dist = float(np.min(goal_distances))
        self._episode_stats["min_goal_distance"] = min(
            self._episode_stats["min_goal_distance"], min_dist
        )
        self._episode_stats["final_goal_distance"] = min_dist

        # 8. Check termination conditions
        done, reason = self._check_termination()
        if done and not self._episode_terminated:
            self._episode_terminated = True
            self._termination_reason = reason

        # 9. Get perception observations
        observations = None
        if self._perception_system is not None:
            observations = self._perception_system.observe(self.state)

        # 10. Build comprehensive info dict
        # Ensure collisions key always exists
        if "collisions" not in info_dict:
            info_dict["collisions"] = []

        info_dict.update(
            {
                "step": self._step_count,
                "done": self._episode_terminated,
                "termination_reason": self._termination_reason,
                "episode_stats": self._episode_stats.copy(),
                "observations": observations if observations is not None else [],
            }
        )

        # 11. Log for Visualization
        if self.enable_logging:
            self.log_frame(info=info_dict)

        # 12. Call Render Hook
        if self.render_hook:
            self.render_hook(self.state, info_dict)

        return self.state, info_dict

    def log_frame(self, info: Optional[Dict[str, Any]] = None) -> None:
        """
        Append the current swarm state to the run history.

        Args:
            info: Optional info dict containing collision events, etc.
        """
        if self.current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")

        frame = SimulationFrame(
            state=self.state.clone(), info=info.copy() if info else {}
        )
        self.current_run.frames.append(frame)

    def save_run(self, output_path: Path) -> None:
        """
        Persist the current simulation run to disk for offline visualization.

        Args:
            output_path: Path where the SimulationRun will be saved
        """
        if self.current_run is None:
            raise RuntimeError("No run to save. Call start_run() first.")

        # Helper function to make info dict JSON serializable
        def serialize_info(info: dict) -> dict:
            """Convert info dict to JSON-serializable format"""
            serialized = {}
            for key, value in info.items():
                if key == "collisions":
                    # Convert CollisionInfo objects to dicts
                    serialized[key] = [
                        {
                            "drone_id": int(
                                c.drone_id
                            ),  # Convert numpy int to Python int
                            "collision_type": c.collision_type,
                            "normal_vector": c.normal_vector.tolist(),
                            "contact_point": c.contact_point.tolist(),
                            "penetration_depth": float(c.penetration_depth),
                            "rebound_velocity": c.rebound_velocity.tolist(),
                            "new_position": c.new_position.tolist(),
                        }
                        for c in value
                    ]
                elif key == "observations":
                    # Convert SensorReading objects to dicts
                    serialized[key] = [
                        {
                            "ranges": obs.ranges.tolist(),
                            "hits": obs.hits.tolist(),
                            "neighbor_vectors": obs.neighbor_vectors.tolist(),
                            "metadata": {
                                k: v.tolist() if hasattr(v, "tolist") else v
                                for k, v in obs.metadata.items()
                            },
                        }
                        for obs in value
                    ]
                else:
                    # Other fields are already serializable
                    serialized[key] = value
            return serialized

        # Convert to serializable format (JSON)
        data = {
            "metadata": self.current_run.metadata,
            "frames": [
                {
                    "state": {
                        "t": frame.state.t,
                        "pos": frame.state.pos.tolist(),
                        "vel": frame.state.vel.tolist(),
                        "acc": frame.state.acc.tolist(),
                        "ids": frame.state.ids.tolist(),
                        "goals": frame.state.goals.tolist(),
                        "metadata": frame.state.metadata,
                    },
                    "info": serialize_info(frame.info),
                }
                for frame in self.current_run.frames
            ],
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def reset(self, randomize: bool = False, seed: Optional[int] = None) -> SwarmState:
        """
        Reset the simulator to initial conditions for a new episode.

        Args:
            randomize: If True, randomize initial positions/velocities. If False, reset to exact initial state.
            seed: Random seed for randomization (only used if randomize=True)

        Returns:
            The reset SwarmState
        """
        if self._initial_state is None:
            raise RuntimeError("No initial state stored. Call start_run() first.")

        if randomize:
            # Randomize initial state
            rng = np.random.RandomState(seed)
            N = self._initial_state.pos.shape[0]

            # Randomize positions within some bounds
            # For now, just add small perturbations to initial positions
            positions = self._initial_state.pos.copy() + rng.randn(N, 3) * 0.5

            # Randomize velocities (small random velocities)
            velocities = rng.randn(N, 3) * 0.1

            # Goals are always required
            goals = self._initial_state.goals.copy()

            self.state = SwarmState.from_initial_positions(
                positions=positions, ids=self._initial_state.ids.copy(), goals=goals
            )
            self.state.vel = velocities
        else:
            # Reset to exact initial state
            self.state = self._initial_state.clone()

        # Reset episode management
        self._step_count = 0
        self._episode_terminated = False
        self._termination_reason = None

        # Reset episode statistics
        self._episode_stats = {
            "collision_count": 0,
            "min_goal_distance": float("inf"),
            "final_goal_distance": float("inf"),
            "total_steps": 0,
        }

        # Clear current run frames (but keep the run object)
        if self.current_run is not None:
            self.current_run.frames = []
            self.log_frame(info={"event": "episode_reset"})

        return self.state

    def _validate_actions(self, actions: np.ndarray) -> np.ndarray:
        """
        Validate and optionally clip actions to ensure they are safe.

        Args:
            actions: Action array, shape (N, 3)

        Returns:
            Validated/clipped action array
        """
        N = self.state.pos.shape[0] if self.state.pos is not None else 0

        # Check shape
        expected_shape = (N, 3)
        if actions.shape != expected_shape:
            raise ValueError(
                f"Invalid action shape. Expected {expected_shape}, got {actions.shape}"
            )

        # Check for NaN or Inf
        if np.any(~np.isfinite(actions)):
            warnings.warn("Actions contain NaN or Inf values. Clipping to zero.")
            actions = np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0)

        # Clip to max acceleration if configured
        if self.config.max_acceleration is not None:
            action_mags = np.linalg.norm(actions, axis=1)
            exceeded = action_mags > self.config.max_acceleration
            if np.any(exceeded):
                warnings.warn(
                    f"{np.sum(exceeded)} action(s) exceed max acceleration. Clipping to {self.config.max_acceleration} m/s^2."
                )
                # Normalize and scale
                scale = np.minimum(
                    1.0, self.config.max_acceleration / (action_mags + 1e-12)
                )
                actions = actions * scale[:, np.newaxis]

        return actions

    def _check_termination(self) -> tuple[bool, Optional[str]]:
        """
        Check if episode should terminate based on various conditions.

        Returns:
            (done, reason) tuple where done is True if episode should end,
            and reason is a string describing why
        """
        # Already terminated (e.g., by collision)
        if self._episode_terminated:
            return True, self._termination_reason

        # Check timeout
        if self._step_count >= self.config.max_steps:
            return True, "timeout"

        # Check success (all drones within goal threshold)
        goal_distances = np.linalg.norm(self.state.pos - self.state.goals, axis=1)
        if np.all(goal_distances <= self.config.goal_threshold):
            return True, "success"

        # Check out-of-bounds
        if hasattr(self.environment, "bounds"):
            bounds = self.environment.bounds
            if callable(bounds):
                bounds = bounds()

            if bounds is not None:
                x_min, x_max, y_min, y_max, z_min, z_max = bounds
                pos = self.state.pos

                out_of_bounds = (
                    np.any(pos[:, 0] < x_min)
                    or np.any(pos[:, 0] > x_max)
                    or np.any(pos[:, 1] < y_min)
                    or np.any(pos[:, 1] > y_max)
                    or np.any(pos[:, 2] < z_min)
                    or np.any(pos[:, 2] > z_max)
                )

                if out_of_bounds:
                    return True, "out_of_bounds"

        return False, None
