"""
Core simulator interface skeleton.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .state import SwarmState

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
    ) -> None:
        self.delta_t = delta_t
        self.collision_system = collision_system
        self.render_hook = render_hook
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
        pass

    def step(self, actions: np.ndarray) -> tuple[SwarmState, dict]:
        """
        Advance the simulation by one tick and return the new state and info dict.
        
        Returns:
            Tuple of (updated SwarmState, info dict with collision events, etc.)
        
        Core Simulation team: Apply kinematics, collision detection, and update state.
        """
        pass

    def log_frame(self, info: Optional[Dict[str, Any]] = None) -> None:
        """
        Append the current swarm state to the run history.
        
        Args:
            info: Optional info dict containing collision events, etc.
        
        Core Simulation team: Append a SimulationFrame to self.current_run.
        """
        pass

    def save_run(self, output_path: Path) -> None:
        """
        Persist the current simulation run to disk for offline visualization.
        
        Args:
            output_path: Path where the SimulationRun will be saved
        
        Core Simulation team: Serialize self.current_run to disk (e.g., JSON, pickle).
        """
        pass
