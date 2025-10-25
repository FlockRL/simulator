"""Configuration data models using Pydantic for validation.

Each team can extend these configs with additional fields as needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from pydantic import BaseModel, Field


class SimulationConfig(BaseModel):
    """
    Core simulation parameters.
    
    delta_t: Simulation time step [s] (default: 1/240 for 240 Hz)
    max_steps: Maximum number of simulation steps
    """
    delta_t: float = Field(default=1.0 / 240.0, gt=0.0, description="Time step in seconds")
    max_steps: int = Field(default=1000, gt=0, description="Maximum simulation steps")


class EnvironmentConfig(BaseModel):
    """
    Environment generation parameters.
    
    bounds: (x_min, x_max, y_min, y_max, z_min, z_max) defining the 3D box
    seed: Random seed for reproducible environment generation
    
    Obstacles team: Add additional fields for wall count, gate specifications,
    clutter density, etc. as described in your design doc.
    """
    bounds: Tuple[float, float, float, float, float, float] = Field(
        default=(-5.0, 5.0, -5.0, 5.0, 0.0, 5.0),
        description="Simulation bounds (x_min, x_max, y_min, y_max, z_min, z_max)"
    )
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


class CollisionConfig(BaseModel):
    """
    Collision detection and response parameters.
    
    restitution: Coefficient of restitution (1.0 = perfectly elastic, 0.0 = inelastic)
    enable_collisions: Master switch to enable/disable all collision handling
    
    Collision team: Add flags for individual collision types (wall, gate, clutter)
    if you want fine-grained control.
    """
    restitution: float = Field(default=1.0, ge=0.0, le=1.0, description="Coefficient of restitution")
    enable_collisions: bool = Field(default=True, description="Enable collision detection and response")


class VisualizationConfig(BaseModel):
    """
    Visualization and rendering parameters.
    
    render_mode: "offline" for post-processing, "online" for real-time (future)
    fps: Frames per second for visualization playback
    save_path: Path to save visualization logs/videos
    
    Visualization team: Add camera settings, viewport options, etc. as needed.
    """
    render_mode: str = Field(default="offline", description="Rendering mode (offline/online)")
    fps: int = Field(default=60, gt=0, description="Frames per second for playback")
    save_path: Optional[Path] = Field(default=None, description="Path to save visualization output")

