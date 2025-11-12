"""Environment-level specification models."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator

from .obstacles import ClutterSpec, GateSpec, WallSpec


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

    name: str = Field(..., description="Environment name/identifier")
    description: Optional[str] = Field(
        default=None,
        description="Human-readable description of this environment",
    )
    bounds: Tuple[float, float, float, float, float, float] = Field(
        default=(-5.0, 5.0, -5.0, 5.0, 0.0, 5.0),
        description="Environment bounds (x_min, x_max, y_min, y_max, z_min, z_max)",
    )
    random_seed: Optional[int] = Field(
        default=None,
        description="Seed used for random sampling to keep environments reproducible",
    )
    obstacles: List[Union[WallSpec, GateSpec, ClutterSpec]] = Field(
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
        cls, obstacles: List[Union[WallSpec, GateSpec, ClutterSpec]]
    ) -> List[Union[WallSpec, GateSpec, ClutterSpec]]:
        ids = [obs.id for obs in obstacles]
        if len(ids) != len(set(ids)):
            raise ValueError("Obstacle template IDs must be unique")
        return obstacles

    @field_validator("obstacles")
    @classmethod
    def validate_gate_references(
        cls, obstacles: List[Union[WallSpec, GateSpec, ClutterSpec]]
    ) -> List[Union[WallSpec, GateSpec, ClutterSpec]]:
        gate_ids = {obs.id for obs in obstacles if isinstance(obs, GateSpec)}
        referenced_gate_ids: set[str] = set()

        for obs in obstacles:
            if isinstance(obs, WallSpec) and obs.gate_id is not None:
                if obs.gate_id not in gate_ids:
                    raise ValueError(
                        f"Wall {obs.id} references non-existent gate {obs.gate_id}"
                    )
                referenced_gate_ids.add(obs.gate_id)

        unused_gates = gate_ids - referenced_gate_ids
        if unused_gates:
            unused = ", ".join(sorted(unused_gates))
            raise ValueError(f"Gate templates unused by any wall: {unused}")

        return obstacles


__all__ = ["SpawnZoneSpec", "EnvironmentSpec"]
