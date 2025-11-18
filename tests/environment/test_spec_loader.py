"""Tests for EnvironmentSpecLoader file loading functionality."""

import json
from pathlib import Path

import pytest

from flockrl_sim.environment import EnvironmentSpecLoader
from flockrl_sim.environment.spec_models.obstacles import ClutterSpec


class TestSpecLoaderPresets:
    """Test preset loading functionality."""

    def test_list_presets(self):
        """Test listing available presets."""
        loader = EnvironmentSpecLoader()
        presets = loader.list_presets()

        assert "simple" in presets
        assert "medium" in presets
        assert "complex" in presets
        assert "empty" in presets
        assert "manual_only" in presets
        assert "random_only" in presets

    def test_load_simple_preset(self):
        """Test loading a simple preset."""
        loader = EnvironmentSpecLoader()
        spec = loader.load_preset("simple")

        assert spec.name == "simple"
        assert spec.random_seed == 42
        assert any(isinstance(obs, ClutterSpec) and obs.random for obs in spec.obstacles)

    def test_load_nonexistent_preset(self):
        """Test that loading non-existent preset raises error."""
        loader = EnvironmentSpecLoader()
        with pytest.raises(FileNotFoundError, match="not found"):
            loader.load_preset("does_not_exist")


class TestSpecLoaderPathHandling:
    """Test file path handling."""

    def test_load_accepts_suffix_and_path(self):
        """Test that load accepts both string and Path objects."""
        loader = EnvironmentSpecLoader()

        spec_with_suffix = loader.load("simple.json")
        assert spec_with_suffix.name == "simple"

        spec_from_path = loader.load(Path("medium.json"))
        assert spec_from_path.name == "medium"

    def test_load_from_custom_path(self, tmp_path):
        """Test loading from arbitrary file path."""
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


class TestSpecLoaderErrorHandling:
    """Test error handling in spec loader."""

    def test_load_invalid_json(self, tmp_path):
        """Test that invalid JSON raises appropriate error."""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("{ invalid json }")

        loader = EnvironmentSpecLoader()
        with pytest.raises(ValueError, match="Invalid JSON"):
            loader.load_from_path(invalid_file)

    def test_load_validation_error_preserves_details(self, tmp_path):
        """Test that Pydantic validation errors are preserved."""
        invalid_spec = tmp_path / "invalid_spec.json"
        invalid_spec.write_text(json.dumps({
            "name": "invalid",
            "description": "Missing required fields",
            "bounds": [-5, 5, -5, 5, 0, 5],
        }))

        loader = EnvironmentSpecLoader()
        with pytest.raises(ValueError) as exc_info:
            loader.load_from_path(invalid_spec)

        error_msg = str(exc_info.value)
        assert "Validation failed" in error_msg
        assert "Field required" in error_msg or "required" in error_msg.lower()
