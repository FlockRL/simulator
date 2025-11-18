from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple
from pydantic import BaseModel, Field

class SimulationConfig(BaseModel):
    delta_t: float = Field(default=1.0 / 240.0, gt=0.0, description="Time step in seconds")
    max_steps: int = Field(default=1000, gt=0, description="Maximum simulation steps")
    goal_threshold: float = Field(default=0.5, gt=0.0, description="Distance threshold for goal success (meters)")
    max_acceleration: Optional[float] = Field(default=None, description="Maximum allowed acceleration magnitude (m/s^2), None for no limit")
    terminate_on_collision: bool = Field(default=True, description="Whether to terminate episode immediately on collision")

class EnvironmentConfig(BaseModel):
    spec: Optional[str] = Field(default=None, description="Environment spec name or path to JSON file")
    bounds: Optional[Tuple[float, float, float, float, float, float]] = Field(default=None, description="Simulation bounds (overrides spec)")
    seed: Optional[int] = Field(default=None, description="Random seed (overrides spec)")

class CollisionConfig(BaseModel):
    restitution: float = Field(default=1.0, ge=0.0, le=1.0, description="Coefficient of restitution")
    enable_collisions: bool = Field(default=True, description="Enable collision detection and response")

class VisualizationConfig(BaseModel):
    render_mode: str = Field(default="offline", description="Rendering mode (offline/online)")
    fps: int = Field(default=60, gt=0, description="Frames per second for playback")
    save_path: Optional[Path] = Field(default=None, description="Path to save visualization output")

