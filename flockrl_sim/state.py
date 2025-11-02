"""
Shared swarm state container definitions.

This is the "single source of truth" for the simulation state.
All teams read from SwarmState; only Core writes to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SwarmState:
    """
    Canonical drone state bag-of-data.
    
    As specified in the design documents:
    - t: Simulation time [s]
    - pos: Position array, shape (N, 3) [m]
    - vel: Velocity array, shape (N, 3) [m/s]
    - acc: Acceleration array, shape (N, 3) [m/s^2]
    - ids: Drone ID array, shape (N,) [int]
    - metadata: Optional additional data (collision events, etc.)
    """

    t: float = 0.0
    pos: np.ndarray | None = None  # shape (N, 3) [m]
    vel: np.ndarray | None = None  # shape (N, 3) [m/s]
    acc: np.ndarray | None = None  # shape (N, 3) [m/s^2]
    ids: np.ndarray | None = None  # shape (N,) [int]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_initial_positions(cls, positions: np.ndarray, ids: np.ndarray) -> SwarmState:
        """
        Create a SwarmState from initial positions with zero velocities and accelerations.
        
        Args:
            positions: Initial positions, shape (N, 3)
            ids: Drone IDs, shape (N,)
        
        Returns:
            Initialized SwarmState with zero velocities and accelerations
        """
        N = positions.shape[0]
        velocities = np.zeros((N, 3))
        accelerations = np.zeros((N, 3))
        return SwarmState(pos=positions, vel=velocities, acc=accelerations, ids=ids)

    def clone(self) -> SwarmState:
        """
        Create a deep copy of this SwarmState.
        
        Useful for collision detection (testing proposed states) and logging.
        
            Core Simulation team: Implement state cloning logic here.
        """
        return SwarmState(
            t = self.t,
            pos=self.pos.copy() if self.pos is not None else None,
            vel=self.vel.copy() if self.vel is not None else None,
            acc=self.acc.copy() if self.acc is not None else None,
            ids=self.ids.copy() if self.ids is not None else None,
            metadata=self.metadata.copy()
        )
