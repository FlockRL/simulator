"""Obstacle specification models."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .random_values import (
    OptionalScalarValue,
    PartialVector3Value,
    ScalarValue,
    Vector3Value,
    contains_random_value,
    validate_positive_scalar,
)


class RandomizableModel(BaseModel):
    """Base model that supports optional randomization."""

    random: bool = Field(
        default=False,
        description="Whether this obstacle should sample random values",
    )
    count: int = Field(
        default=1,
        ge=1,
        description="Number of instances to generate when random is true",
    )

    def _randomizable_components(self) -> List[Any]:
        """Return fields that can contain random configs."""
        return []

    @model_validator(mode="after")
    def _validate_randomization(self):
        if not self.random and self.count != 1:
            raise ValueError("Non-random obstacles must have count == 1")

        if not self.random:
            for component in self._randomizable_components():
                if contains_random_value(component):
                    raise ValueError(
                        "Obstacle uses random values but 'random' is not set to true"
                    )
        return self


class ObstacleSpec(RandomizableModel):
    """Base obstacle specification."""

    id: str = Field(..., description="Unique identifier for this obstacle template")
    type: str = Field(..., description="Obstacle type: 'wall', 'gate', or 'clutter'")


class WallSpec(ObstacleSpec):
    """Specification for a wall obstacle."""

    type: Literal["wall"] = "wall"
    position: Vector3Value = Field(
        ...,
        description="(x, y, z) position; components can be deterministic or random",
    )
    orientation: Optional[Vector3Value] = Field(
        default=None,
        description="(roll, pitch, yaw) orientation; defaults to (0, 0, 0)",
    )
    length: ScalarValue = Field(..., description="Wall length in meters")
    height: ScalarValue = Field(..., description="Wall height in meters")
    thickness: ScalarValue = Field(
        default=0.1,
        description="Wall thickness in meters",
    )
    gate_id: Optional[str] = Field(
        default=None,
        description="ID of gate template embedded in this wall",
    )

    _length_positive = field_validator("length")(
        lambda cls, v: validate_positive_scalar(v, "Wall length")
    )
    _height_positive = field_validator("height")(
        lambda cls, v: validate_positive_scalar(v, "Wall height")
    )
    _thickness_positive = field_validator("thickness")(
        lambda cls, v: validate_positive_scalar(v, "Wall thickness")
    )

    def _randomizable_components(self) -> List[Any]:
        components: List[Any] = [self.position, self.length, self.height, self.thickness]
        if self.orientation is not None:
            components.append(self.orientation)
        return components


class GateSpec(ObstacleSpec):
    """Specification for a gate (opening in a wall)."""

    type: Literal["gate"] = "gate"
    position: Optional[PartialVector3Value] = Field(
        default=None,
        description=(
            "Gate position. Components set to null inherit the parent wall position."
        ),
    )
    orientation: Optional[PartialVector3Value] = Field(
        default=None,
        description=(
            "Gate orientation. Components set to null inherit the parent wall orientation."
        ),
    )
    width: ScalarValue = Field(..., description="Gate width in meters")
    height: ScalarValue = Field(..., description="Gate height in meters")
    frame_thickness: ScalarValue = Field(
        default=0.05,
        description="Frame thickness in meters",
    )

    _width_positive = field_validator("width")(
        lambda cls, v: validate_positive_scalar(v, "Gate width")
    )
    _height_positive = field_validator("height")(
        lambda cls, v: validate_positive_scalar(v, "Gate height")
    )
    _frame_positive = field_validator("frame_thickness")(
        lambda cls, v: validate_positive_scalar(v, "Gate frame thickness")
    )

    def _randomizable_components(self) -> List[Any]:
        components: List[Any] = [self.width, self.height, self.frame_thickness]
        if self.position is not None:
            components.append(self.position)
        if self.orientation is not None:
            components.append(self.orientation)
        return components

    @model_validator(mode="after")
    def _enforce_single_gate_template(self):
        if self.count != 1:
            raise ValueError(
                "Gate specs must have count == 1; walls duplicate gates automatically"
            )
        return self


class ClutterSpec(ObstacleSpec):
    """Specification for clutter geometry."""

    type: Literal["clutter"] = "clutter"
    subtype: Literal["rectangular_prism"] = Field(
        default="rectangular_prism",
        description="Clutter subtype",
    )
    position: Vector3Value = Field(..., description="Clutter position in meters")
    orientation: Optional[Vector3Value] = Field(
        default=None,
        description="Orientation; defaults to (0, 0, 0)",
    )
    length: ScalarValue = Field(..., description="Length in meters")
    width: ScalarValue = Field(..., description="Width in meters")
    height: ScalarValue = Field(..., description="Height in meters")

    _length_positive = field_validator("length")(
        lambda cls, v: validate_positive_scalar(v, "Clutter length")
    )
    _width_positive = field_validator("width")(
        lambda cls, v: validate_positive_scalar(v, "Clutter width")
    )
    _height_positive = field_validator("height")(
        lambda cls, v: validate_positive_scalar(v, "Clutter height")
    )

    def _randomizable_components(self) -> List[Any]:
        components: List[Any] = [self.position, self.length, self.width, self.height]
        if self.orientation is not None:
            components.append(self.orientation)
        return components


__all__ = [
    "RandomizableModel",
    "ObstacleSpec",
    "WallSpec",
    "GateSpec",
    "ClutterSpec",
]
