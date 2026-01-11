"""
Reward function base class for the FlockRL Gymnasium environment.

Users should subclass RewardFunction to define their own reward shaping.
"""

from typing import Any, Dict

import numpy as np

from .state import SwarmState


class RewardFunction:
    """
    Base class for reward functions.
    
    Subclass this to define custom reward shaping for your training.
    Implement the `compute()` method (and optionally `reset()` if you need state tracking).
    """

    def reset(self, state: SwarmState) -> None:
        """
        Called when environment is reset.
        
        Override this to initialize any state needed for reward computation
        (e.g., previous goal distance for computing deltas).
        
        Default implementation does nothing - only override if you need state tracking.
        
        Args:
            state: Initial state after reset
        """
        pass

    def compute(
        self, state: SwarmState, action: np.ndarray, sim_info: Dict[str, Any]
    ) -> np.ndarray:
        """
        Compute rewards for all drones in the current step.
        
        Must be implemented by subclasses.
        
        Args:
            state: Current simulation state with N drones
            action: Actions that were taken (after clipping), shape (N, 3)
            sim_info: Dictionary with simulation information including:
                - termination_reason: How episode ended (if done)
                - episode_stats: Statistics about the episode
                - collisions: List of collision events
                
        Returns:
            Reward array of shape (N,) - one reward per drone
        """
        raise NotImplementedError("Subclasses must implement compute()")


__all__ = ["RewardFunction"]
