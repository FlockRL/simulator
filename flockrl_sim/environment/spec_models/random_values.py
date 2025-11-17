from __future__ import annotations
from typing import Annotated, Any, List, Optional, Tuple, Union
from pydantic import BaseModel, BeforeValidator, field_validator

class UniformRandomConfig(BaseModel):
    """Uniformly distributed scalar value."""

    uniform: Tuple[float, float]

    @field_validator("uniform")
    @classmethod
    def validate_uniform(cls, value: Tuple[float, float]) -> Tuple[float, float]:
        low, high = value
        if low > high:
            raise ValueError("Uniform range min must not be more than max")
        return value

class DiscreteRandomConfig(BaseModel):
    """Discrete set of scalar choices."""
    discrete: List[float]

    @field_validator("discrete")
    @classmethod
    def validate_discrete(cls, values: List[float]) -> List[float]:
        if not values:
            raise ValueError("Discrete value list must not be empty")
        return [float(v) for v in values]

ScalarValue = Annotated[
    Union[float, UniformRandomConfig, DiscreteRandomConfig],
    BeforeValidator(lambda x: float(x) if isinstance(x, int) else x)
]
Vector3Value = Tuple[ScalarValue, ScalarValue, ScalarValue]

# Allows for inheritance of parent values. Used only for gates.
PartialVector3Value = Tuple[Optional[ScalarValue], Optional[ScalarValue], Optional[ScalarValue]]

def contains_random_value(value: Any) -> bool:
    """Recursively check if the value contains any random configuration."""
    if isinstance(value, (UniformRandomConfig, DiscreteRandomConfig)):
        return True
    if isinstance(value, (tuple, list)):
        return any(contains_random_value(v) for v in value)
    return False


def validate_positive_scalar(value: ScalarValue, field_name: str) -> ScalarValue:
    """Ensure scalar (or random config) represents positive values."""
    if isinstance(value, float):
        if value <= 0:
            raise ValueError(f"{field_name} must be positive")
    elif isinstance(value, UniformRandomConfig):
        if any(v <= 0 for v in value.uniform):
            raise ValueError(f"{field_name} uniform range must be positive")
    elif isinstance(value, DiscreteRandomConfig):
        if any(v <= 0 for v in value.discrete):
            raise ValueError(f"All discrete choices for {field_name} must be positive")

    return value


__all__ = [
    "UniformRandomConfig",
    "DiscreteRandomConfig",
    "ScalarValue",
    "Vector3Value",
    "PartialVector3Value",
    "contains_random_value",
    "validate_positive_scalar",
]
