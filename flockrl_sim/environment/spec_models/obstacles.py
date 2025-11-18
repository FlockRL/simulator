from __future__ import annotations
from typing import Any, List, Union, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from .random_values import PartialVector3Value, ScalarValue, Vector3Value, contains_random_value, validate_positive_scalar

class GateSpec(BaseModel):
    """Gate specification (part of a wall, not a standalone obstacle)."""

    position: PartialVector3Value
    width: ScalarValue
    height: ScalarValue

    @model_validator(mode="after")
    def validate_positive_dimensions(self):
        self.width = validate_positive_scalar(self.width, "Gate width")
        self.height = validate_positive_scalar(self.height, "Gate height")
        return self

class ObstacleSpec(BaseModel):
    """Base obstacle specification that supports optional randomization."""

    id: str
    type: str

    position: Vector3Value
    orientation: Vector3Value

    random: bool = False
    count: int = Field(default=1, ge=1)

    def _randomizable_components(self) -> List[Any]:
        """Return fields that actually contain random configs (uniform or discrete)."""
        return []

    @model_validator(mode="after")
    def _validate_randomization(self):
        if not self.random and self.count != 1:
            raise ValueError("Non-random obstacles must have count == 1")

        if not self.random and self._randomizable_components():
            raise ValueError("Obstacle uses random values but 'random' is not set to true")
        
        return self

class WallSpec(ObstacleSpec):
    """Specification for a wall obstacle."""

    type: Literal["wall"] = "wall"
    length: ScalarValue
    height: ScalarValue
    thickness: ScalarValue
    gates: List[GateSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_positive_dimensions(self):
        self.length = validate_positive_scalar(self.length, "Wall length")
        self.height = validate_positive_scalar(self.height, "Wall height")
        self.thickness = validate_positive_scalar(self.thickness, "Wall thickness")
        return self

    def _randomizable_components(self) -> List[Any]:
        """Return fields that actually contain random configs (uniform or discrete)."""
        all_components = [
            self.position, self.orientation, self.length, self.height, self.thickness,
            *[comp for gate in self.gates for comp in (gate.position, gate.width, gate.height)],
        ]
        return [c for c in all_components if contains_random_value(c)]

class ClutterSpec(ObstacleSpec):
    """Specification for clutter geometry."""

    type: Literal["clutter"] = "clutter"
    subtype: Union[Literal["rectangular_prism"]] # Add more subtypes here
    length: ScalarValue
    width: ScalarValue
    height: ScalarValue

    @model_validator(mode="after")
    def validate_positive_dimensions(self):
        self.length = validate_positive_scalar(self.length, "Clutter length")
        self.width = validate_positive_scalar(self.width, "Clutter width")
        self.height = validate_positive_scalar(self.height, "Clutter height")
        return self

    def _randomizable_components(self) -> List[Any]:
        """Return fields that actually contain random configs (uniform or discrete)."""
        all_components = [self.position, self.orientation, self.length, self.width, self.height]
        return [c for c in all_components if contains_random_value(c)]

__all__ = [
    "ObstacleSpec",
    "WallSpec",
    "GateSpec",
    "ClutterSpec",
]
