"""Unit tests for environment specs, loader, and builder."""

import json
from math import hypot

import pytest

from flockrl_sim.environment import EnvironmentSpecLoader, EnvironmentBuilder
from flockrl_sim.environment.spec_models.environment import EnvironmentSpec
from flockrl_sim.environment.spec_models.obstacles import WallSpec, GateSpec, ClutterSpec
from flockrl_sim.environment.spec_models.random_values import UniformRandomConfig
from flockrl_sim.environment.obstacles_types import Wall, RectangularPrism
from flockrl_sim.environment.obstacles import SPAWN_CLEARANCE_METERS
from flockrl_sim.environment.validation import check_overlap


class TestEnvironmentSpec:
    """Test EnvironmentSpec Pydantic model."""

    def test_minimal_spec(self):
        spec = EnvironmentSpec(name="test", description="Test environment")
        assert spec.name == "test"
        assert spec.bounds == (-5.0, 5.0, -5.0, 5.0, 0.0, 5.0)
        assert spec.random_seed is None
        assert len(spec.obstacles) == 0

    def test_spec_with_obstacles(self):
        spec = EnvironmentSpec(
            name="test_manual",
            obstacles=[
                WallSpec(
                    id="wall1",
                    position=(1.0, 0.0, 0.0),
                    length=5.0,
                    height=3.0,
                    gate_id="gate1",
                ),
                GateSpec(
                    id="gate1",
                    position=(1.0, 0.0, 1.0),
                    width=1.5,
                    height=1.5,
                ),
                ClutterSpec(
                    id="clutter1",
                    position=(2.0, 2.0, 0.0),
                    length=0.5,
                    width=0.5,
                    height=0.8,
                ),
            ],
        )
        assert len(spec.obstacles) == 3
        assert isinstance(spec.obstacles[0], WallSpec)
        assert isinstance(spec.obstacles[1], GateSpec)
        assert isinstance(spec.obstacles[2], ClutterSpec)

    def test_invalid_bounds(self):
        with pytest.raises(ValueError, match="x_min.*must be less than x_max"):
            EnvironmentSpec(
                name="invalid",
                bounds=(5.0, -5.0, -5.0, 5.0, 0.0, 5.0),
            )

    def test_duplicate_obstacle_ids(self):
        with pytest.raises(ValueError, match="Obstacle template IDs must be unique"):
            EnvironmentSpec(
                name="duplicate_ids",
                obstacles=[
                    WallSpec(id="wall1", position=(0, 0, 0), length=1, height=1),
                    WallSpec(id="wall1", position=(1, 0, 0), length=1, height=1),
                ],
            )

    def test_invalid_gate_reference(self):
        with pytest.raises(ValueError, match="references non-existent gate"):
            EnvironmentSpec(
                name="bad_gate",
                obstacles=[
                    WallSpec(
                        id="wall1",
                        position=(0.0, 0.0, 0.0),
                        length=5.0,
                        height=3.0,
                        gate_id="missing_gate",
                    )
                ],
            )

    def test_unused_gate_template(self):
        with pytest.raises(ValueError, match="Gate templates unused"):
            EnvironmentSpec(
                name="unused_gate",
                obstacles=[
                    GateSpec(
                        id="gate_template",
                        width=1.0,
                        height=1.0,
                        frame_thickness=0.05,
                    ),
                    WallSpec(
                        id="wall1",
                        position=(0.0, 0.0, 0.0),
                        length=2.0,
                        height=1.0,
                    ),
                ],
            )

    def test_random_flag_requires_matching_count(self):
        with pytest.raises(ValueError, match="count == 1"):
            WallSpec(
                id="wall_random",
                position=(0.0, 0.0, 0.0),
                length=2.0,
                height=1.0,
                random=False,
                count=2,
            )

    def test_random_flag_required_for_uniform_values(self):
        with pytest.raises(ValueError, match="random values but 'random' is not set"):
            WallSpec(
                id="wall_uniform",
                position=(UniformRandomConfig(uniform=(0.0, 1.0)), 0.0, 0.0),
                length=2.0,
                height=1.0,
            )

    def test_gate_template_forces_single_count(self):
        with pytest.raises(ValueError, match="Gate specs must have count == 1"):
            GateSpec(
                id="gate_template",
                random=True,
                count=2,
                width=1.0,
                height=1.0,
                frame_thickness=0.05,
            )


class TestEnvironmentSpecLoader:
    """Test EnvironmentSpecLoader functionality."""

    def test_loader_initialization(self):
        loader = EnvironmentSpecLoader()
        assert loader.specs_dir.exists()
        assert loader.specs_dir.name == "specs"

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

    def test_load_medium_preset(self):
        loader = EnvironmentSpecLoader()
        spec = loader.load_preset("medium")

        assert spec.name == "medium"
        assert spec.random_seed == 123

    def test_load_nonexistent_preset(self):
        loader = EnvironmentSpecLoader()
        with pytest.raises(FileNotFoundError, match="not found"):
            loader.load_preset("does_not_exist")

    def test_smart_load_preset(self):
        loader = EnvironmentSpecLoader()
        spec = loader.load("simple")
        assert spec.name == "simple"

    def test_smart_load_path(self, tmp_path):
        spec_data = {
            "name": "temp_spec",
            "description": "Temporary test spec",
            "bounds": [-5, 5, -5, 5, 0, 5],
            "obstacles": [],
        }
        temp_file = tmp_path / "temp_spec.json"
        with open(temp_file, "w") as handle:
            json.dump(spec_data, handle)

        loader = EnvironmentSpecLoader()
        spec = loader.load(str(temp_file))
        assert spec.name == "temp_spec"

    def test_load_from_path(self, tmp_path):
        spec_data = {
            "name": "custom",
            "bounds": [-10, 10, -10, 10, 0, 10],
            "obstacles": [
                {
                    "id": "wall1",
                    "type": "wall",
                    "position": [0, 0, 0],
                    "length": 5,
                    "height": 3,
                    "thickness": 0.1,
                }
            ],
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
        assert "gate1" in obstacle_ids

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

        spawn_positions = []
        if spec.spawn_zones:
            for pos in (spec.spawn_zones.start_position, spec.spawn_zones.goal_position):
                if pos:
                    spawn_positions.append(pos)

        for obs in env.obstacles:
            for spawn in spawn_positions:
                distance = hypot(obs.position[0] - spawn[0], obs.position[1] - spawn[1])
                assert distance >= SPAWN_CLEARANCE_METERS

        for i, obs1 in enumerate(env.obstacles):
            for obs2 in env.obstacles[i + 1 :]:
                if isinstance(obs1, Wall) and obs1.gate_id == obs2.id:
                    continue
                if isinstance(obs2, Wall) and obs2.gate_id == obs1.id:
                    continue
                assert not check_overlap(obs1, obs2)

    def test_random_wall_gate_suffixes_and_inheritance(self):
        spec = EnvironmentSpec(
            name="suffix_demo",
            random_seed=2024,
            bounds=(-5.0, 5.0, -5.0, 5.0, 0.0, 5.0),
            obstacles=[
                GateSpec(
                    id="gate_template",
                    position=(None, 0.0, 1.0),
                    width=1.0,
                    height=1.0,
                    frame_thickness=0.05,
                ),
                WallSpec(
                    id="wall_template",
                    random=True,
                    count=2,
                    position=(
                        UniformRandomConfig(uniform=(-2.0, 2.0)),
                        UniformRandomConfig(uniform=(-1.5, 1.5)),
                        0.0,
                    ),
                    length=4.0,
                    height=2.5,
                    thickness=0.2,
                    gate_id="gate_template",
                ),
            ],
        )

        env = EnvironmentBuilder.from_spec(spec).build()
        walls = [obs for obs in env.obstacles if isinstance(obs, Wall)]
        assert len(walls) == 2

        for idx, wall in enumerate(sorted(walls, key=lambda w: w.id)):
            expected_gate_id = f"gate_template_{idx}"
            assert wall.gate_id == expected_gate_id
            gate = env.get_obstacle_by_id(expected_gate_id)
            assert gate is not None
            assert gate.position[0] == pytest.approx(wall.position[0])

    def test_random_clutter_generation_count(self):
        spec = EnvironmentSpec(
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
                        0.0,
                    ),
                    length=UniformRandomConfig(uniform=(0.5, 1.0)),
                    width=UniformRandomConfig(uniform=(0.5, 1.0)),
                    height=UniformRandomConfig(uniform=(0.5, 1.0)),
                )
            ],
        )

        env = EnvironmentBuilder.from_spec(spec).build()
        clutters = [obs for obs in env.obstacles if isinstance(obs, RectangularPrism)]
        assert len(clutters) == 4
