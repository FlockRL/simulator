from __future__ import annotations
import json
from pathlib import Path
from typing import Union
from flockrl_sim.environment.spec_models.environment import EnvironmentSpec

class EnvironmentSpecLoader:
    def __init__(self):
        self.specs_dir = Path(__file__).parent / "specs"
        if not self.specs_dir.exists():
            raise FileNotFoundError(f"Specs directory not found at {self.specs_dir}")

    def load_preset(self, name: str) -> EnvironmentSpec:
        name = f"{name}.json" if not name.endswith(".json") else name
        preset_path = self.specs_dir / name
        if not preset_path.exists():
            available = ', '.join(self.list_presets())
            raise FileNotFoundError(f"Preset '{name}' not found. Available: {available}")
        return self._load_from_file(preset_path)

    def load_from_path(self, path: Union[str, Path]) -> EnvironmentSpec:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Environment spec file not found: {path}")
        return self._load_from_file(path)

    def load(self, name_or_path: Union[str, Path]) -> EnvironmentSpec:
        """Load environment spec from preset name (e.g., 'simple') or file path.

        Automatically detects whether input is a preset or path based on structure.
        """
        path = Path(name_or_path)

        # If it's already a Path object or has multiple parts, treat as file path
        if isinstance(name_or_path, Path) or len(path.parts) > 1:
            return self.load_from_path(path)

        # If it has a file extension, treat as file path
        if path.suffix:
            return self.load_from_path(path)

        # If it exists as a file, treat as file path
        if path.exists() and path.is_file():
            return self.load_from_path(path)

        # Otherwise, treat as preset name
        return self.load_preset(str(name_or_path))

    def list_presets(self) -> list[str]:
        if not self.specs_dir.exists():
            return []
        return sorted([f.stem for f in self.specs_dir.glob("*.json")
                      if f.stem not in ["README", "schema"]])

    def _load_from_file(self, path: Path) -> EnvironmentSpec:
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {path}: {e}")
        try:
            return EnvironmentSpec(**data)
        except Exception as e:
            raise ValueError(f"Validation failed for {path}: {e}")
