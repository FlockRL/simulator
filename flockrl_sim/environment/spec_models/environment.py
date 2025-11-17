from __future__ import annotations
from typing import List, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator
from .obstacles import ClutterSpec, WallSpec

class SpawnZoneSpec(BaseModel):
    """Specification for spawn zones (start and goal positions)."""

    start_position: Optional[Tuple[float, float, float]] = Field(
        default=None,
        description="Exact start position (if None, randomly generated)",
    )
    goal_position: Optional[Tuple[float, float, float]] = Field(
        default=None,
        description="Exact goal position (if None, randomly generated)",
    )
    start_zone_bounds: Optional[Tuple[float, float, float, float, float, float]] = Field(
        default=None,
        description=(
            "Bounds for random start position generation "
            "(x_min, x_max, y_min, y_max, z_min, z_max)"
        ),
    )
    goal_zone_bounds: Optional[Tuple[float, float, float, float, float, float]] = Field(
        default=None,
        description=(
            "Bounds for random goal position generation "
            "(x_min, x_max, y_min, y_max, z_min, z_max)"
        ),
    )


class EnvironmentSpec(BaseModel):
    """Complete environment specification supporting manual and random placement."""

    name: str
    description: Optional[str] = None
    bounds: Tuple[float, float, float, float, float, float] = Field(
        default=(-5.0, 5.0, -5.0, 5.0, 0.0, 5.0),
        description="Environment bounds (x_min, x_max, y_min, y_max, z_min, z_max)",
    )
    random_seed: Optional[int] = Field(
        default=None,
        description="Seed used for random sampling to keep environments reproducible",
    )
    obstacles: List[Union[WallSpec, ClutterSpec]] = Field(
        default_factory=list,
        description="List of obstacle templates (manual or random)",
    )
    spawn_zones: Optional[SpawnZoneSpec] = Field(
        default=None,
        description="Start and goal position configuration",
    )

    @field_validator("bounds")
    @classmethod
    def validate_bounds(
        cls, bounds: Tuple[float, float, float, float, float, float]
    ) -> Tuple[float, float, float, float, float, float]:
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        if x_min >= x_max:
            raise ValueError(f"x_min ({x_min}) must be less than x_max ({x_max})")
        if y_min >= y_max:
            raise ValueError(f"y_min ({y_min}) must be less than y_max ({y_max})")
        if z_min >= z_max:
            raise ValueError(f"z_min ({z_min}) must be less than z_max ({z_max})")
        return bounds

    @field_validator("obstacles")
    @classmethod
    def validate_unique_ids(
        cls, obstacles: List[Union[WallSpec, ClutterSpec]]
    ) -> List[Union[WallSpec, ClutterSpec]]:
        ids = [obs.id for obs in obstacles]
        if len(ids) != len(set(ids)):
            raise ValueError("Obstacle template IDs must be unique")
        return obstacles


__all__ = ["SpawnZoneSpec", "EnvironmentSpec"]
