"""
Goal management for drone policy deployment.

Handles waypoints, goal reaching detection, and coordinate transforms between
simulator coordinates and real-world arena coordinates.
"""

import numpy as np
from typing import List, Tuple, Optional


class GoalHandler:
    """Manage goals and waypoints for drone navigation."""
    
    def __init__(
        self,
        goals: List[Tuple[float, float, float]],
        goal_tolerance: float = 0.3,
        arena_origin: Optional[Tuple[float, float, float]] = None,
        arena_scale: float = 1.0
    ):
        """
        Initialize goal handler.
        
        Args:
            goals: List of goal positions [(x, y, z), ...] in simulator coordinates
            goal_tolerance: Distance threshold for reaching goal (meters)
            arena_origin: Real-world coordinates of simulator origin (x, y, z)
            arena_scale: Scale factor from simulator to real world
        """
        self.goals = [np.array(g, dtype=np.float32) for g in goals]
        self.goal_tolerance = goal_tolerance
        self.arena_origin = np.array(arena_origin if arena_origin else [0, 0, 0], dtype=np.float32)
        self.arena_scale = arena_scale
        
        self.current_goal_idx = 0
        self.goals_reached = []
        self.start_time = None
        
    def get_current_goal(self) -> np.ndarray:
        """Get current goal position in simulator coordinates."""
        if self.current_goal_idx >= len(self.goals):
            # Return last goal if all completed
            return self.goals[-1]
        return self.goals[self.current_goal_idx]
    
    def get_current_goal_real_world(self) -> np.ndarray:
        """Get current goal in real-world coordinates."""
        sim_goal = self.get_current_goal()
        return self.sim_to_real(sim_goal)
    
    def sim_to_real(self, sim_pos: np.ndarray) -> np.ndarray:
        """Convert simulator coordinates to real-world coordinates."""
        return self.arena_origin + sim_pos * self.arena_scale
    
    def real_to_sim(self, real_pos: np.ndarray) -> np.ndarray:
        """Convert real-world coordinates to simulator coordinates."""
        return (real_pos - self.arena_origin) / self.arena_scale
    
    def check_goal_reached(self, current_pos: np.ndarray) -> bool:
        """
        Check if current goal is reached.
        
        Args:
            current_pos: Current position in simulator coordinates
            
        Returns:
            True if goal reached
        """
        if self.current_goal_idx >= len(self.goals):
            return True  # All goals completed
        
        goal = self.get_current_goal()
        distance = np.linalg.norm(current_pos - goal)
        
        if distance <= self.goal_tolerance:
            self.goals_reached.append({
                "goal_idx": self.current_goal_idx,
                "goal_pos": goal.copy(),
                "reached_pos": current_pos.copy(),
                "distance": float(distance)
            })
            self.current_goal_idx += 1
            return True
        
        return False
    
    def compute_goal_vector(self, current_pos: np.ndarray) -> np.ndarray:
        """
        Compute vector from current position to goal.
        
        Args:
            current_pos: Current position in simulator coordinates
            
        Returns:
            Goal vector (dx, dy, dz)
        """
        goal = self.get_current_goal()
        return goal - current_pos
    
    def compute_goal_distance(self, current_pos: np.ndarray) -> float:
        """
        Compute Euclidean distance to current goal.
        
        Args:
            current_pos: Current position in simulator coordinates
            
        Returns:
            Distance to goal in meters
        """
        goal_vector = self.compute_goal_vector(current_pos)
        return float(np.linalg.norm(goal_vector))
    
    def is_complete(self) -> bool:
        """Check if all goals have been reached."""
        return self.current_goal_idx >= len(self.goals)
    
    def get_progress(self) -> dict:
        """Get progress statistics."""
        return {
            "current_goal_idx": self.current_goal_idx,
            "total_goals": len(self.goals),
            "goals_reached": len(self.goals_reached),
            "complete": self.is_complete(),
            "current_goal": self.get_current_goal().tolist(),
            "reached_history": self.goals_reached
        }
    
    def reset(self):
        """Reset to first goal."""
        self.current_goal_idx = 0
        self.goals_reached = []
        self.start_time = None


class SingleGoalHandler(GoalHandler):
    """Simplified handler for single goal navigation."""
    
    def __init__(
        self,
        goal: Tuple[float, float, float],
        goal_tolerance: float = 0.3,
        arena_origin: Optional[Tuple[float, float, float]] = None,
        arena_scale: float = 1.0
    ):
        """
        Initialize with single goal.
        
        Args:
            goal: Goal position (x, y, z) in simulator coordinates
            goal_tolerance: Distance threshold for reaching goal (meters)
            arena_origin: Real-world coordinates of simulator origin
            arena_scale: Scale factor from simulator to real world
        """
        super().__init__(
            goals=[goal],
            goal_tolerance=goal_tolerance,
            arena_origin=arena_origin,
            arena_scale=arena_scale
        )


class WaypointHandler(GoalHandler):
    """Handler for sequential waypoint navigation."""
    
    def __init__(
        self,
        waypoints: List[Tuple[float, float, float]],
        goal_tolerance: float = 0.3,
        arena_origin: Optional[Tuple[float, float, float]] = None,
        arena_scale: float = 1.0,
        loop: bool = False
    ):
        """
        Initialize waypoint handler.
        
        Args:
            waypoints: List of waypoints [(x, y, z), ...]
            goal_tolerance: Distance threshold for reaching each waypoint
            arena_origin: Real-world coordinates of simulator origin
            arena_scale: Scale factor from simulator to real world
            loop: If True, return to first waypoint after completing all
        """
        super().__init__(
            goals=waypoints,
            goal_tolerance=goal_tolerance,
            arena_origin=arena_origin,
            arena_scale=arena_scale
        )
        self.loop = loop
    
    def check_goal_reached(self, current_pos: np.ndarray) -> bool:
        """Check if current waypoint reached, handle looping."""
        reached = super().check_goal_reached(current_pos)
        
        if reached and self.loop and self.current_goal_idx >= len(self.goals):
            # Loop back to first waypoint
            self.current_goal_idx = 0
            
        return reached


# Preset goal configurations for common scenarios
PRESET_GOALS = {
    "hover": SingleGoalHandler(
        goal=(0.0, 0.0, 0.5),
        goal_tolerance=0.2
    ),
    
    "forward_1m": SingleGoalHandler(
        goal=(1.0, 0.0, 0.5),
        goal_tolerance=0.3
    ),
    
    "square_1m": WaypointHandler(
        waypoints=[
            (1.0, 0.0, 0.5),
            (1.0, 1.0, 0.5),
            (0.0, 1.0, 0.5),
            (0.0, 0.0, 0.5)
        ],
        goal_tolerance=0.3,
        loop=False
    ),
    
    "figure_eight": WaypointHandler(
        waypoints=[
            (0.5, 0.0, 0.5),
            (1.0, 0.5, 0.5),
            (0.5, 1.0, 0.5),
            (0.0, 0.5, 0.5),
            (-0.5, 1.0, 0.5),
            (-1.0, 0.5, 0.5),
            (-0.5, 0.0, 0.5),
            (0.0, -0.5, 0.5)
        ],
        goal_tolerance=0.25,
        loop=True
    )
}


def create_goal_handler(preset: str, **kwargs) -> GoalHandler:
    """
    Create a goal handler from preset name.
    
    Args:
        preset: Name of preset configuration or "custom"
        **kwargs: Override parameters for preset or custom goal specification
        
    Returns:
        Configured GoalHandler instance
    """
    if preset == "custom":
        if "goal" in kwargs:
            return SingleGoalHandler(**kwargs)
        elif "goals" in kwargs or "waypoints" in kwargs:
            goals = kwargs.pop("goals", kwargs.pop("waypoints", []))
            return WaypointHandler(waypoints=goals, **kwargs)
        else:
            raise ValueError("Custom goal requires 'goal' or 'goals'/'waypoints' parameter")
    
    if preset not in PRESET_GOALS:
        raise ValueError(f"Unknown preset '{preset}'. Available: {list(PRESET_GOALS.keys())}")
    
    handler = PRESET_GOALS[preset]
    
    # Apply any overrides
    if "goal_tolerance" in kwargs:
        handler.goal_tolerance = kwargs["goal_tolerance"]
    if "arena_origin" in kwargs:
        handler.arena_origin = np.array(kwargs["arena_origin"], dtype=np.float32)
    if "arena_scale" in kwargs:
        handler.arena_scale = kwargs["arena_scale"]
    
    return handler
