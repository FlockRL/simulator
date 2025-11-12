"""Random helper types and validators for environment specifications."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator


Number = Union[int, float]


class UniformRandomConfig(BaseModel):
    """Uniformly distributed scalar value."""

    uniform: Tuple[float, float] = Field(
        ...,
        description="Inclusive (min, max) bounds for a uniform distribution",
    )

    @field_validator("uniform")
    @classmethod
    def validate_uniform(cls, value: Tuple[float, float]) -> Tuple[float, float]:
        low, high = value
        if low >= high:
            raise ValueError("Uniform range min must be less than max")
        return value


class DiscreteRandomConfig(BaseModel):
    """Discrete set of scalar choices."""

    discrete: List[float] = Field(
        ...,
        description="List of discrete values to choose from uniformly",
    )

    @field_validator("discrete")
    @classmethod
    def validate_discrete(cls, values: List[float]) -> List[float]:
        if not values:
            raise ValueError("Discrete value list must not be empty")
        return values


ScalarValue = Union[Number, UniformRandomConfig, DiscreteRandomConfig]
OptionalScalarValue = Union[ScalarValue, None]
Vector3Value = Tuple[ScalarValue, ScalarValue, ScalarValue]
PartialVector3Value = Tuple[
    OptionalScalarValue, OptionalScalarValue, OptionalScalarValue
]


def contains_random_value(value: Any) -> bool:
    """Return True if the value contains any random configuration."""
    if value is None:
        return False
    if isinstance(value, (UniformRandomConfig, DiscreteRandomConfig)):
        return True
    if isinstance(value, (tuple, list)):
        return any(contains_random_value(component) for component in value)
    return False


def validate_positive_scalar(value: ScalarValue, field_name: str) -> ScalarValue:
    """Ensure scalar (or random config) represents positive values."""
    if isinstance(value, (int, float)):
        if value <= 0:
            raise ValueError(f"{field_name} must be positive")
        return float(value)

    if isinstance(value, UniformRandomConfig):
        low, high = value.uniform
        if low <= 0 or high <= 0:
            raise ValueError(f"{field_name} uniform range must be positive")
        return value

    if isinstance(value, DiscreteRandomConfig):
        if any(choice <= 0 for choice in value.discrete):
            raise ValueError(f"All discrete choices for {field_name} must be positive")
        return value

    return value


__all__ = [
    "Number",
    "UniformRandomConfig",
    "DiscreteRandomConfig",
    "ScalarValue",
    "OptionalScalarValue",
    "Vector3Value",
    "PartialVector3Value",
    "contains_random_value",
    "validate_positive_scalar",
]
