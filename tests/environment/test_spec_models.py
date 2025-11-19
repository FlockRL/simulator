"""Tests for EnvironmentSpec Pydantic models and validation."""

import pytest

from flockrl_sim.environment.spec_models.environment import EnvironmentSpec
from flockrl_sim.environment.spec_models.obstacles import (
    WallSpec,
    ClutterSpec,
    GateSpec,
)
from flockrl_sim.environment.spec_models.random_values import UniformRandomConfig


DEFAULT_SPEC_KWARGS = {
    "description": "Unit test environment",
    "bounds": (-5.0, 5.0, -5.0, 5.0, 0.0, 5.0),
    "random_seed": 123,
    "start_position": (-4.0, 0.0, 0.0),
    "goal_position": (4.0, 0.0, 0.0),
}


def spec_kwargs(**overrides):
    """Return base EnvironmentSpec kwargs with overrides."""
    data = dict(DEFAULT_SPEC_KWARGS)
    data.update(overrides)
    return data


class TestEnvironmentSpecValidation:
    """Test EnvironmentSpec Pydantic model validation."""

    def test_minimal_valid_spec(self):
        """Test creation of minimal valid spec."""
        spec = EnvironmentSpec(**spec_kwargs(name="test"))
        assert spec.name == "test"
        assert spec.bounds == DEFAULT_SPEC_KWARGS["bounds"]
        assert spec.random_seed == DEFAULT_SPEC_KWARGS["random_seed"]
        assert len(spec.obstacles) == 0

    def test_spec_with_obstacles(self):
        """Test spec with various obstacle types."""
        spec = EnvironmentSpec(
            **spec_kwargs(
                name="test_manual",
                obstacles=[
                    WallSpec(
                        id="wall1",
                        position=(1.0, 0.0, 0.0),
                        orientation=(0.0, 0.0, 0.0),
                        length=5.0,
                        height=3.0,
                        thickness=0.1,
                        gates=[
                            GateSpec(
                                position=(1.0, 0.0, 0.5),
                                width=1.5,
                                height=1.5,
                            )
                        ],
                    ),
                    ClutterSpec(
                        id="clutter1",
                        position=(2.0, 2.0, 0.0),
                        orientation=(0.0, 0.0, 0.0),
                        subtype="rectangular_prism",
                        length=0.5,
                        width=0.5,
                        height=0.8,
                    ),
                ],
            )
        )
        assert len(spec.obstacles) == 2
        assert isinstance(spec.obstacles[0], WallSpec)
        assert isinstance(spec.obstacles[1], ClutterSpec)
        assert len(spec.obstacles[0].gates) == 1

    def test_invalid_bounds(self):
        """Test that invalid bounds are rejected."""
        with pytest.raises(ValueError, match="must have min < max"):
            EnvironmentSpec(
                **spec_kwargs(
                    name="invalid",
                    bounds=(5.0, -5.0, -5.0, 5.0, 0.0, 5.0),
                )
            )

    def test_duplicate_obstacle_ids(self):
        """Test that duplicate obstacle IDs are rejected."""
        with pytest.raises(ValueError, match="Obstacle template IDs must be unique"):
            EnvironmentSpec(
                **spec_kwargs(
                    name="duplicate_ids",
                    obstacles=[
                        WallSpec(
                            id="wall1",
                            position=(0, 0, 0),
                            orientation=(0, 0, 0),
                            length=1,
                            height=1,
                            thickness=0.1,
                        ),
                        WallSpec(
                            id="wall1",
                            position=(1, 0, 0),
                            orientation=(0, 0, 0),
                            length=1,
                            height=1,
                            thickness=0.1,
                        ),
                    ],
                )
            )


class TestRandomObstacleValidation:
    """Test validation rules for random obstacles."""

    def test_random_flag_requires_matching_count(self):
        """Test that non-random obstacles must have count=1."""
        with pytest.raises(ValueError, match="count == 1"):
            WallSpec(
                id="wall_random",
                position=(0.0, 0.0, 0.0),
                orientation=(0.0, 0.0, 0.0),
                length=2.0,
                height=1.0,
                thickness=0.1,
                random=False,
                count=2,
            )

    def test_random_flag_required_for_uniform_values(self):
        """Test that uniform values require random=True."""
        with pytest.raises(ValueError, match="random values but 'random' is not set"):
            WallSpec(
                id="wall_uniform",
                position=(UniformRandomConfig(uniform=(0.0, 1.0)), 0.0, 0.0),
                orientation=(0.0, 0.0, 0.0),
                length=2.0,
                height=1.0,
                thickness=0.1,
            )
