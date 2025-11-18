"""Unit tests for environment specs, loader, and builder."""

import json
from math import hypot

import pytest

from flockrl_sim.environment import EnvironmentSpecLoader, EnvironmentBuilder
from flockrl_sim.environment.spec_models.environment import EnvironmentSpec
from flockrl_sim.environment.spec_models.obstacles import WallSpec, ClutterSpec, GateSpec
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


class TestEnvironmentSpec:
    """Test EnvironmentSpec Pydantic model."""

    def test_minimal_spec(self):
        spec = EnvironmentSpec(**spec_kwargs(name="test"))
        assert spec.name == "test"
        assert spec.bounds == DEFAULT_SPEC_KWARGS["bounds"]
        assert spec.random_seed == DEFAULT_SPEC_KWARGS["random_seed"]
        assert spec.start_position == DEFAULT_SPEC_KWARGS["start_position"]
        assert spec.goal_position == DEFAULT_SPEC_KWARGS["goal_position"]
        assert len(spec.obstacles) == 0

    def test_spec_with_obstacles(self):
        spec = EnvironmentSpec(**spec_kwargs(
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
        ))
        assert len(spec.obstacles) == 2
        assert isinstance(spec.obstacles[0], WallSpec)
        assert isinstance(spec.obstacles[1], ClutterSpec)
        assert len(spec.obstacles[0].gates) == 1

    def test_invalid_bounds(self):
        with pytest.raises(ValueError, match="must have min < max"):
            EnvironmentSpec(**spec_kwargs(
                name="invalid",
                bounds=(5.0, -5.0, -5.0, 5.0, 0.0, 5.0),
            ))

    def test_duplicate_obstacle_ids(self):
        with pytest.raises(ValueError, match="Obstacle template IDs must be unique"):
            EnvironmentSpec(**spec_kwargs(
                name="duplicate_ids",
                obstacles=[
                    WallSpec(id="wall1", position=(0, 0, 0), orientation=(0, 0, 0), length=1, height=1, thickness=0.1),
                    WallSpec(id="wall1", position=(1, 0, 0), orientation=(0, 0, 0), length=1, height=1, thickness=0.1),
                ],
            ))

    # Removed: test_invalid_gate_reference - gates are now inline, can't reference missing gates

    # Removed: test_unused_gate_template - gates are now inline, can't be unused

    def test_random_flag_requires_matching_count(self):
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
        with pytest.raises(ValueError, match="random values but 'random' is not set"):
            WallSpec(
                id="wall_uniform",
                position=(UniformRandomConfig(uniform=(0.0, 1.0)), 0.0, 0.0),
                orientation=(0.0, 0.0, 0.0),
                length=2.0,
                height=1.0,
                thickness=0.1,
            )

    # Removed: test_gate_template_forces_single_count - gates are now inline, no separate gate specs


class TestEnvironmentSpecLoader:
    """Test EnvironmentSpecLoader functionality."""

    def test_list_presets(self):
        loader = EnvironmentSpecLoader()
        presets = loader.list_presets()

        assert "simple" in presets
        assert "medium" in presets
        assert "complex" in presets
        assert "empty" in presets
        assert "manual_only" in presets
        assert "random_only" in presets

    def test_load_simple_preset(self):
        loader = EnvironmentSpecLoader()
        spec = loader.load_preset("simple")

        assert spec.name == "simple"
        assert spec.random_seed == 42
        assert any(isinstance(obs, ClutterSpec) and obs.random for obs in spec.obstacles)

    def test_load_nonexistent_preset(self):
        loader = EnvironmentSpecLoader()
        with pytest.raises(FileNotFoundError, match="not found"):
            loader.load_preset("does_not_exist")

    def test_load_from_path(self, tmp_path):
        spec_data = {
            "name": "custom",
            "description": "Custom spec loaded from arbitrary path",
            "bounds": [-10, 10, -10, 10, 0, 10],
            "obstacles": [
                {
                    "id": "wall1",
                    "type": "wall",
                    "position": [0, 0, 0],
                    "orientation": [0, 0, 0],
                    "length": 5,
                    "height": 3,
                    "thickness": 0.1,
                }
            ],
            "random_seed": 99,
            "start_position": [-8, 0, 0],
            "goal_position": [8, 0, 0],
        }
        custom_file = tmp_path / "custom.json"
        with open(custom_file, "w") as handle:
            json.dump(spec_data, handle)

        loader = EnvironmentSpecLoader()
        spec = loader.load_from_path(custom_file)

        assert spec.name == "custom"
        assert spec.bounds == (-10, 10, -10, 10, 0, 10)
        assert len(spec.obstacles) == 1

    def test_load_invalid_json(self, tmp_path):
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("{ invalid json }")

        loader = EnvironmentSpecLoader()
        with pytest.raises(ValueError, match="Invalid JSON"):
            loader.load_from_path(invalid_file)

    def test_load_validation_error_preserves_details(self, tmp_path):
        """Test that Pydantic validation errors are preserved with field-level details."""
        invalid_spec = tmp_path / "invalid_spec.json"
        # Missing required fields: random_seed, start_position, goal_position
        invalid_spec.write_text(json.dumps({
            "name": "invalid",
            "description": "Missing required fields",
            "bounds": [-5, 5, -5, 5, 0, 5],
            # Missing: random_seed, start_position, goal_position
        }))

        loader = EnvironmentSpecLoader()
        with pytest.raises(ValueError) as exc_info:
            loader.load_from_path(invalid_spec)

        error_msg = str(exc_info.value)
        assert "Validation failed" in error_msg
        # Pydantic error should mention the missing fields
        assert "Field required" in error_msg or "required" in error_msg.lower()


class TestEnvironmentBuilder:
    """Test EnvironmentBuilder with specs."""

    def test_build_from_empty_spec(self):
        loader = EnvironmentSpecLoader()
        spec = loader.load("empty")

        env = EnvironmentBuilder.from_spec(spec).build()
        assert len(env.obstacles) == 0
        assert env.bounds == spec.bounds

    def test_build_from_manual_only_spec(self):
        loader = EnvironmentSpecLoader()
        spec = loader.load("manual_only")

        env = EnvironmentBuilder.from_spec(spec).build()
        obstacle_ids = {obs.id for obs in env.obstacles}
        assert "wall1" in obstacle_ids

        wall = next(obs for obs in env.obstacles if isinstance(obs, Wall))
        # Gate ID format: {wall_id}_gate_{index}
        expected_gate_id = f"{wall.id}_gate_0"
        assert len(wall.gate_ids) == 1
        assert wall.gate_ids[0] == expected_gate_id
        assert wall.gate_ids[0] in obstacle_ids
        assert env.get_obstacle_by_id(wall.gate_ids[0]) is not None

    def test_build_from_random_spec_reproducibility(self):
        loader = EnvironmentSpecLoader()
        spec = loader.load("random_only")

        env1 = EnvironmentBuilder.from_spec(spec).build()
        env2 = EnvironmentBuilder.from_spec(spec).build()

        assert len(env1.obstacles) == len(env2.obstacles)
        for obs1, obs2 in zip(env1.obstacles, env2.obstacles):
            assert obs1.id == obs2.id
            assert obs1.position == obs2.position

    def test_random_generation_respects_spawn_clearance_and_collisions(self):
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

        for i, obs1 in enumerate(env.obstacles):
            for obs2 in env.obstacles[i + 1 :]:
                # Skip if one is the other's gate
                if (isinstance(obs1, Wall) and obs2.id in obs1.linked_gate_ids()) or \
                   (isinstance(obs2, Wall) and obs1.id in obs2.linked_gate_ids()):
                    continue
                assert not check_overlap(obs1, obs2)

    def test_random_wall_gate_suffixes_and_inheritance(self):
        spec = EnvironmentSpec(**spec_kwargs(
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
        ))

        env = EnvironmentBuilder.from_spec(spec).build()
        walls = [obs for obs in env.obstacles if isinstance(obs, Wall)]
        assert len(walls) == 2

        for wall in sorted(walls, key=lambda w: w.id):
            expected_gate_id = f"{wall.id}_gate_0"
            assert len(wall.gate_ids) == 1
            assert wall.gate_ids[0] == expected_gate_id
            gate = env.get_obstacle_by_id(expected_gate_id)
            assert gate is not None
            assert gate.position[0] == pytest.approx(wall.position[0])
            assert isinstance(gate, Gate)
            assert gate.thickness == pytest.approx(wall.thickness)

    def test_manual_walls_with_gates_get_unique_gate_ids(self):
        """Test that multiple walls with the same gate spec get unique gate IDs."""
        spec = EnvironmentSpec(**spec_kwargs(
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
        ))

        env = EnvironmentBuilder.from_spec(spec).build()
        walls = sorted([obs for obs in env.obstacles if isinstance(obs, Wall)], key=lambda w: w.id)
        assert len(walls) == 2

        assert all(wall.gate_ids for wall in walls)
        gate_ids = {wall.gate_ids[0] for wall in walls}
        assert len(gate_ids) == len(walls), "Each wall should have a unique gate ID"

        for wall in walls:
            expected_gate_id = f"{wall.id}_gate_0"
            assert len(wall.gate_ids) == 1
            assert wall.gate_ids[0] == expected_gate_id
            gate = env.get_obstacle_by_id(expected_gate_id)
            assert gate is not None
            assert gate.thickness == pytest.approx(wall.thickness)

    def test_wall_with_multiple_inline_gates_links_all_gate_ids(self):
        """Ensure each inline gate is linked back to its parent wall."""
        spec = EnvironmentSpec(**spec_kwargs(
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
        ))

        env = EnvironmentBuilder.from_spec(spec).build()
        walls = [obs for obs in env.obstacles if isinstance(obs, Wall)]
        gates = [obs for obs in env.obstacles if isinstance(obs, Gate)]

        assert len(walls) == 1
        wall = walls[0]
        assert len(wall.gate_ids) == 2
        assert wall.gate_ids == (f"{wall.id}_gate_0", f"{wall.id}_gate_1")
        assert wall.linked_gate_ids() == wall.gate_ids

        assert len(gates) == 2
        assert {gate.id for gate in gates} == set(wall.gate_ids)
        for gate in gates:
            assert gate.thickness == pytest.approx(wall.thickness)
            assert gate.orientation == wall.orientation
            assert env.get_obstacle_by_id(gate.id) is gate

    def test_random_clutter_generation_count(self):
        spec = EnvironmentSpec(**spec_kwargs(
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
        ))

        env = EnvironmentBuilder.from_spec(spec).build()
        clutters = [obs for obs in env.obstacles if isinstance(obs, RectangularPrism)]
        assert len(clutters) == 4

    def test_spawn_positions_persist_on_environment(self):
        """Resolved spawn positions should be exposed on the built environment."""
        spec = EnvironmentSpec(**spec_kwargs(name="spawn_persist"))

        env = EnvironmentBuilder.from_spec(spec).build()

        assert env.start_position == spec.start_position
        assert env.goal_position == spec.goal_position

    def test_placement_failure_raises_error(self):
        """Test that placement failures raise ValueError instead of silently failing."""
        spec = EnvironmentSpec(**spec_kwargs(
            name="impossible_placement",
            random_seed=42,
            bounds=(-1.0, 1.0, -1.0, 1.0, 0.0, 2.0),  # Very small bounds
            obstacles=[
                WallSpec(
                    id="wall_template",
                    random=True,
                    count=100,  # Impossible to fit 100 walls in tiny bounds
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
        ))

        with pytest.raises(ValueError) as exc_info:
            EnvironmentBuilder.from_spec(spec).build()

        assert "Unable to place wall" in str(exc_info.value)
        assert "without collisions" in str(exc_info.value)
        assert "attempts" in str(exc_info.value)
