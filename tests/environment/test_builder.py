"""Tests for EnvironmentBuilder construction from specs."""

from math import hypot

import pytest
import numpy as np

from flockrl_sim.environment import EnvironmentSpecLoader, EnvironmentBuilder
from flockrl_sim.environment.spec_models.environment import EnvironmentSpec
from flockrl_sim.environment.spec_models.obstacles import (
    WallSpec,
    ClutterSpec,
    GateSpec,
)
from flockrl_sim.environment.spec_models.random_values import UniformRandomConfig
from flockrl_sim.environment.obstacles_types import Gate, Wall, RectangularPrism
from flockrl_sim.environment.obstacles import SPAWN_CLEARANCE_METERS
from flockrl_sim.environment.validation import check_overlap


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


class TestBasicBuilding:
    """Test basic environment building from specs."""

    def test_build_from_empty_spec(self):
        """Test building environment from empty spec."""
        loader = EnvironmentSpecLoader()
        spec = loader.load("empty")

        env = EnvironmentBuilder.from_spec(spec).build()
        assert len(env.obstacles) == 0
        assert env.bounds == spec.bounds

    def test_build_from_manual_only_spec(self):
        """Test building environment with manual obstacles only."""
        loader = EnvironmentSpecLoader()
        spec = loader.load("manual_only")

        env = EnvironmentBuilder.from_spec(spec).build()
        obstacle_ids = {obs.id for obs in env.obstacles}
        assert "wall1" in obstacle_ids

        wall = next(obs for obs in env.obstacles if isinstance(obs, Wall))
        expected_gate_id = f"{wall.id}_gate_0"
        assert len(wall.gate_ids) == 1
        assert wall.gate_ids[0] == expected_gate_id
        assert env.get_obstacle_by_id(wall.gate_ids[0]) is not None


class TestRandomGeneration:
    """Test random obstacle generation."""

    def test_random_reproducibility(self):
        """Test that random generation is reproducible with same seed."""
        loader = EnvironmentSpecLoader()
        spec = loader.load("random_only")

        env1 = EnvironmentBuilder.from_spec(spec).build()
        env2 = EnvironmentBuilder.from_spec(spec).build()

        assert len(env1.obstacles) == len(env2.obstacles)
        for obs1, obs2 in zip(env1.obstacles, env2.obstacles):
            assert obs1.id == obs2.id
            assert obs1.position == obs2.position

    def test_random_respects_spawn_clearance(self):
        """Test that random obstacles respect spawn clearance."""
        loader = EnvironmentSpecLoader()
        spec = loader.load("random_only")

        env = EnvironmentBuilder.from_spec(spec).build()

        spawn_positions = [
            pos for pos in (spec.start_position, spec.goal_position) if pos
        ]

        for obs in env.obstacles:
            for spawn in spawn_positions:
                distance = hypot(obs.position[0] - spawn[0], obs.position[1] - spawn[1])
                assert distance >= SPAWN_CLEARANCE_METERS

    def test_random_no_overlaps(self):
        """Test that random obstacles don't overlap each other."""
        loader = EnvironmentSpecLoader()
        spec = loader.load("random_only")

        env = EnvironmentBuilder.from_spec(spec).build()

        for i, obs1 in enumerate(env.obstacles):
            for obs2 in env.obstacles[i + 1 :]:
                # Skip if one is the other's gate
                if (isinstance(obs1, Wall) and obs2.id in obs1.linked_gate_ids()) or (
                    isinstance(obs2, Wall) and obs1.id in obs2.linked_gate_ids()
                ):
                    continue
                assert not check_overlap(obs1, obs2)

    def test_random_clutter_count(self):
        """Test that random clutter respects count parameter."""
        spec = EnvironmentSpec(
            **spec_kwargs(
                name="clutter_demo",
                random_seed=7,
                bounds=(-6.0, 6.0, -6.0, 6.0, 0.0, 4.0),
                obstacles=[
                    ClutterSpec(
                        id="clutter_template",
                        random=True,
                        count=4,
                        position=(
                            UniformRandomConfig(uniform=(-5.0, 5.0)),
                            UniformRandomConfig(uniform=(-5.0, 5.0)),
                            UniformRandomConfig(uniform=(0.5, 3.5)),
                        ),
                        orientation=(0.0, 0.0, 0.0),
                        subtype="rectangular_prism",
                        length=UniformRandomConfig(uniform=(0.5, 1.0)),
                        width=UniformRandomConfig(uniform=(0.5, 1.0)),
                        height=UniformRandomConfig(uniform=(0.5, 1.0)),
                    )
                ],
            )
        )

        env = EnvironmentBuilder.from_spec(spec).build()
        clutters = [obs for obs in env.obstacles if isinstance(obs, RectangularPrism)]
        assert len(clutters) == 4

    def test_random_resamples_when_out_of_bounds(self):
        """Test that random generation resamples when placement violates bounds."""
        spec = EnvironmentSpec(
            **spec_kwargs(
                name="clutter_resample",
                random_seed=42,
                bounds=(-5.0, 5.0, -5.0, 5.0, 0.0, 0.5),
                start_position=(-4.0, 0.0, 0.25),
                goal_position=(4.0, 0.0, 0.25),
                obstacles=[
                    ClutterSpec(
                        id="clutter_template",
                        random=True,
                        count=1,
                        position=(
                            UniformRandomConfig(uniform=(-1.0, 2.0)),
                            UniformRandomConfig(uniform=(-1.0, 2.0)),
                            UniformRandomConfig(uniform=(-1.0, 2.0)),
                        ),
                        orientation=(0.0, 0.0, 0.0),
                        subtype="rectangular_prism",
                        length=0.2,
                        width=0.2,
                        height=0.2,
                    ),
                ],
            )
        )

        env = EnvironmentBuilder.from_spec(spec).build()

        clutters = [obs for obs in env.obstacles if isinstance(obs, RectangularPrism)]
        assert len(clutters) == 1
        clutter = clutters[0]

        # Verify accepted placement is within bounds
        assert clutter.position == pytest.approx(
            (1.6765387031145362, -0.7391835021117515, 0.26576545905581117)
        )


class TestGateGeneration:
    """Test gate generation and linkage."""

    def test_random_wall_gate_inheritance(self):
        """Test that gates inherit properties from their walls."""
        spec = EnvironmentSpec(
            **spec_kwargs(
                name="suffix_demo",
                random_seed=2024,
                bounds=(-5.0, 5.0, -5.0, 5.0, 0.0, 5.0),
                obstacles=[
                    WallSpec(
                        id="wall_template",
                        random=True,
                        count=2,
                        position=(
                            UniformRandomConfig(uniform=(-2.0, 2.0)),
                            UniformRandomConfig(uniform=(-1.5, 1.5)),
                            1.25,
                        ),
                        orientation=(0.0, 0.0, 0.0),
                        length=4.0,
                        height=2.5,
                        thickness=0.2,
                        gates=[
                            GateSpec(
                                position=(None, None, 1.25),
                                width=1.0,
                                height=1.0,
                            )
                        ],
                    ),
                ],
            )
        )

        env = EnvironmentBuilder.from_spec(spec).build()
        walls = [obs for obs in env.obstacles if isinstance(obs, Wall)]
        assert len(walls) == 2

        for wall in sorted(walls, key=lambda w: w.id):
            expected_gate_id = f"{wall.id}_gate_0"
            assert len(wall.gate_ids) == 1
            assert wall.gate_ids[0] == expected_gate_id
            gate = env.get_obstacle_by_id(expected_gate_id)
            assert gate is not None
            assert isinstance(gate, Gate)
            assert gate.thickness == pytest.approx(wall.thickness)

    def test_multiple_gates_per_wall(self):
        """Test walls with multiple inline gates."""
        spec = EnvironmentSpec(
            **spec_kwargs(
                name="multi_gate_wall",
                obstacles=[
                    WallSpec(
                        id="wall_multi",
                        position=(0.0, 0.0, 1.25),
                        orientation=(0.0, 0.0, 0.0),
                        length=4.0,
                        height=2.5,
                        thickness=0.2,
                        gates=[
                            GateSpec(
                                position=(None, 0.0, None),
                                width=1.0,
                                height=1.5,
                            ),
                            GateSpec(
                                position=(1.0, 0.0, None),
                                width=1.0,
                                height=1.5,
                            ),
                        ],
                    ),
                ],
            )
        )

        env = EnvironmentBuilder.from_spec(spec).build()
        walls = [obs for obs in env.obstacles if isinstance(obs, Wall)]
        gates = [obs for obs in env.obstacles if isinstance(obs, Gate)]

        assert len(walls) == 1
        wall = walls[0]
        assert len(wall.gate_ids) == 2
        assert wall.gate_ids == (f"{wall.id}_gate_0", f"{wall.id}_gate_1")

        assert len(gates) == 2
        assert {gate.id for gate in gates} == set(wall.gate_ids)

    def test_manual_walls_get_unique_gate_ids(self):
        """Test that multiple walls with gates get unique IDs."""
        spec = EnvironmentSpec(
            **spec_kwargs(
                name="shared_gate_spec",
                obstacles=[
                    WallSpec(
                        id="wall_a",
                        position=(-2.0, 0.0, 1.0),
                        orientation=(0.0, 0.0, 0.0),
                        length=3.0,
                        height=2.0,
                        thickness=0.2,
                        gates=[
                            GateSpec(
                                position=(None, None, None),
                                width=1.0,
                                height=1.5,
                            )
                        ],
                    ),
                    WallSpec(
                        id="wall_b",
                        position=(2.0, 0.0, 1.0),
                        orientation=(0.0, 0.0, 0.0),
                        length=3.0,
                        height=2.0,
                        thickness=0.2,
                        gates=[
                            GateSpec(
                                position=(None, None, None),
                                width=1.0,
                                height=1.5,
                            )
                        ],
                    ),
                ],
            )
        )

        env = EnvironmentBuilder.from_spec(spec).build()
        walls = sorted(
            [obs for obs in env.obstacles if isinstance(obs, Wall)], key=lambda w: w.id
        )
        assert len(walls) == 2

        gate_ids = {wall.gate_ids[0] for wall in walls}
        assert len(gate_ids) == len(walls)


class TestBuilderEdgeCases:
    """Test edge cases in environment building."""

    def test_spawn_positions_persist(self):
        """Test that spawn positions are stored on environment."""
        spec = EnvironmentSpec(**spec_kwargs(name="spawn_persist"))

        env = EnvironmentBuilder.from_spec(spec).build()

        assert env.start_position == spec.start_position
        assert env.goal_position == spec.goal_position

    def test_placement_failure_raises_error(self):
        """Test that impossible placement raises error."""
        spec = EnvironmentSpec(
            **spec_kwargs(
                name="impossible_placement",
                random_seed=42,
                bounds=(-1.0, 1.0, -1.0, 1.0, 0.0, 2.0),
                obstacles=[
                    WallSpec(
                        id="wall_template",
                        random=True,
                        count=100,
                        position=(
                            UniformRandomConfig(uniform=(-0.5, 0.5)),
                            UniformRandomConfig(uniform=(-0.5, 0.5)),
                            1.0,
                        ),
                        orientation=(0.0, 0.0, 0.0),
                        length=2.0,
                        height=2.0,
                        thickness=0.2,
                    ),
                ],
            )
        )

        with pytest.raises(ValueError) as exc_info:
            EnvironmentBuilder.from_spec(spec).build()

        assert "Unable to place wall" in str(exc_info.value)
        assert "without collisions" in str(exc_info.value)
