"""
Core simulator interface skeleton.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .state import SwarmState
from .environment import Environment

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
    """
    Collection of frames associated with a single simulator execution.
    """

    frames: List[SimulationFrame] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CoreSimulator:
    """
    Main simulator class that will handle the simulation loop (update state, apply collisions, etc.).

    Boilerplate class for now, feel free to change as much as you want.
    """

    def __init__(
        self,
        delta_t: float = 1.0 / 60.0, # Defaults to 60 Hz
        collision_system: Optional[CollisionHandler] = None,
        render_hook: Optional[RenderHook] = None,
        environment: Optional[Environment] = None,
    ) -> None:
        self.delta_t = delta_t
        self.collision_system = collision_system
        self.render_hook = render_hook
        self.environment = environment or Environment()  # Default empty environment
        self.state = SwarmState()
        self.current_run: Optional[SimulationRun] = None

    def start_run(self, initial_state: Optional[SwarmState] = None, metadata: Optional[Dict[str, Any]] = None) -> SwarmState:
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
        
        Core Simulation team: Initialize a new SimulationRun and set the initial state.
        """
        if initial_state is None:
            # Create default state with single drone at origin
            positions = np.array([[0.0, 0.0, 1.0]])
            ids = np.array([0])
            self.state = SwarmState.from_initial_positions(positions, ids)
        else:
            self.state = initial_state.clone()

        # Creating a run:
        self.current_run = SimulationRun(
            frames=[],
            metadata=metadata or {}
        )

        # Logging the first frame:
        self.log_frame(info={"event": "run_started"})

        return self.state

    def step(self, actions: np.ndarray) -> tuple[SwarmState, dict]:
        """
        Advance the simulation by one tick and return the new state and info dict.
        
        Returns:
            Tuple of (updated SwarmState, info dict with collision events, etc.)
        
        Core Simulation team: Apply kinematics, collision detection, and update state.
        """
        if self.state.pos is None:
            raise RuntimeError("Simulator state is not initialized. Did you forget to call start_run?")
        
        # 1. Apply actions as accelerations
        self.state.acc = actions
        
        # Calculate proposed new velocities and positions
        proposed_vel = self.state.vel + self.state.acc * self.delta_t
        proposed_pos = self.state.pos + self.state.vel * self.delta_t + 0.5 * self.state.acc * (self.delta_t ** 2)
        proposed_t = self.state.t + self.delta_t

        # 2. Create proposed state for collision checking
        proposed_state = self.state.clone()
        proposed_state.t = proposed_t
        proposed_state.pos = proposed_pos
        proposed_state.vel = proposed_vel
        proposed_state.acc = actions

        info_dict = {}
        final_state = proposed_state

        # 3. Call Collision System (Integration)
        if self.collision_system:
            # The collision system is responsible for detecting collisions
            # and returning a *new state* with corrected positions
            # and rebound velocities.
            final_state, info_dict = self.collision_system(proposed_state)

        # 4. Update master state
        self.state = final_state

        # 5. Log for Visualization (Integration)
        # We log the *final* state *after* collisions are resolved.
        self.log_frame(info=info_dict)

        # 6. Call Render Hook (Integration)
        if self.render_hook:
            self.render_hook(self.state, info_dict)
            
        return self.state, info_dict

    def log_frame(self, info: Optional[Dict[str, Any]] = None) -> None:
        """
        Append the current swarm state to the run history.
        
        Args:
            info: Optional info dict containing collision events, etc.
        
        Core Simulation team: Append a SimulationFrame to self.current_run.
        """
        if self.current_run is None:
            raise RuntimeError("No active run. Call start_run() first.")
        
        frame = SimulationFrame(
            state=self.state.clone(),
            info=info.copy() if info else {}
        )
        self.current_run.frames.append(frame)

    def save_run(self, output_path: Path) -> None:
        """
        Persist the current simulation run to disk for offline visualization.
        
        Args:
            output_path: Path where the SimulationRun will be saved
        
        Core Simulation team: Serialize self.current_run to disk (e.g., JSON, pickle).
        """
        if self.current_run is None:
            raise RuntimeError("No run to save. Call start_run() first.")
        
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
                        "metadata": frame.state.metadata
                    },
                    "info": frame.info
                }
                for frame in self.current_run.frames
            ]
        }
        
        import json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)