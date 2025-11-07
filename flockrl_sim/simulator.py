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
    
        self.current_run = SimulationRun(
            frames=[],
            metadata=metadata or {}
        )
        return self.state

    def step(self, actions: np.ndarray) -> tuple[SwarmState, dict]:
        """
        Advance the simulation by one tick and return the new state and info dict.
        
        Returns:
            Tuple of (updated SwarmState, info dict with collision events, etc.)
        
        Core Simulation team: Apply kinematics, collision detection, and update state.
        """
        # 1. Apply actions as accelerations
        self.state.acc = actions.copy()
        
        # 2. Kinematic integration (Euler method)
        # v(t+dt) = v(t) + a(t) * dt
        # p(t+dt) = p(t) + v(t) * dt + 0.5 * a(t) * dt^2
        
        dt = self.delta_t
        self.state.vel += self.state.acc * dt
        self.state.pos += self.state.vel * dt + 0.5 * self.state.acc * dt**2
        self.state.t += dt
        
        # 3. Apply collision detection (if available)
        info = {}
        if self.collision_system:
            self.state, collision_info = self.collision_system(self.state)
            info.update(collision_info)
        
        # 4. Apply render hook if available
        if self.render_hook:
            self.render_hook(self.state, info)
        
        return self.state, info

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
