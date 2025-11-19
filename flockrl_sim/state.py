"""
Shared swarm state container definitions.

This is the "single source of truth" for the simulation.
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
    - goals: Goal positions array, shape (N, 3) [m] - REQUIRED for all states
    - t: Simulation time [s]
    - pos: Position array, shape (N, 3) [m]
    - vel: Velocity array, shape (N, 3) [m/s]
    - acc: Acceleration array, shape (N, 3) [m/s^2]
    - ids: Drone ID array, shape (N,) [int]
    - metadata: Optional additional data (collision events, etc.)
    """

    goals: np.ndarray  # shape (N, 3) [m] - REQUIRED, must come first (before defaults)
    t: float = 0.0
    pos: np.ndarray | None = None  # shape (N, 3) [m]
    vel: np.ndarray | None = None  # shape (N, 3) [m/s]
    acc: np.ndarray | None = None  # shape (N, 3) [m/s^2]
    ids: np.ndarray | None = None  # shape (N,) [int]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_initial_positions(
        cls, positions: np.ndarray, ids: np.ndarray, goals: np.ndarray
    ) -> SwarmState:
        """
        Create a SwarmState from initial positions with zero velocities and accelerations.

        Args:
            positions: Initial positions, shape (N, 3)
            ids: Drone IDs, shape (N,)
            goals: Goal positions, shape (N, 3)

        Returns:
            Initialized SwarmState with zero velocities and accelerations
        """
        # Copy inputs
        positions = np.array(positions, copy=True)
        ids = np.array(ids, copy=True)
        goals = np.array(goals, copy=True)

        N = positions.shape[0]
        if ids.shape[0] != N:
            raise ValueError(
                "Positions and IDs must have the same number of drones (N)."
            )

        if goals.shape != (N, 3):
            raise ValueError(f"Goals must have shape (N, 3), got {goals.shape}")

        # Initialize velocities and acceleration to zero:
        velocities = np.zeros_like(positions)
        accelerations = np.zeros_like(positions)

        return cls(
            t=0.0,
            pos=positions,
            vel=velocities,
            acc=accelerations,
            ids=ids,
            goals=goals,
            metadata={},
        )

    def clone(self) -> SwarmState:
        """
        Create a deep copy of this SwarmState.

        Useful for collision detection (testing proposed states) and logging.
        """
        return SwarmState(
            t=self.t,
            pos=self.pos.copy() if self.pos is not None else None,
            vel=self.vel.copy() if self.vel is not None else None,
            acc=self.acc.copy() if self.acc is not None else None,
            ids=self.ids.copy() if self.ids is not None else None,
            goals=self.goals.copy(),  # Always required, never None
            metadata=self.metadata.copy(),
        )
