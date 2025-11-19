from __future__ import annotations
from typing import List, Tuple, Union
from pydantic import BaseModel, Field, field_validator
from .obstacles import ClutterSpec, WallSpec
from .random_values import Vector3Value


class EnvironmentSpec(BaseModel):
    """Complete environment specification supporting manual and random placement."""

    name: str
    description: str  # Description of the environment, not very important
    bounds: Tuple[
        float, float, float, float, float, float
    ]  # x_min, x_max, y_min, y_max, z_min, z_max
    random_seed: int  # Seed is to keep environments reproducible
    obstacles: List[Union[WallSpec, ClutterSpec]] = Field(default_factory=list)
    start_position: Vector3Value
    goal_position: Vector3Value

    @field_validator("bounds")
    @classmethod
    def validate_bounds(
        cls, bounds: Tuple[float, float, float, float, float, float]
    ) -> Tuple[float, float, float, float, float, float]:
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        if not (x_min < x_max and y_min < y_max and z_min < z_max):
            raise ValueError(
                f"Invalid bounds: ({x_min}, {x_max}, {y_min}, {y_max}, {z_min}, {z_max}) — must have min < max in each dimension"
            )
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


__all__ = ["EnvironmentSpec"]
