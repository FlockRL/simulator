"""
Unit tests for the OfflineVisualizer class.
"""

import json
import tempfile
from pathlib import Path
import pytest

from flockrl_sim.visualization import OfflineVisualizer


def create_test_log(log_path: Path, num_frames: int = 3, include_obstacles: bool = False) -> None:
    """Helper function to create a test JSON log file."""
    frames = []
    for i in range(num_frames):
        t = i * 0.1
        frames.append({
            "state": {
                "t": t,
                "pos": [[i * 0.1, i * 0.1, i * 0.1], [i * 0.2, i * 0.2, i * 0.2]],
                "vel": [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
                "acc": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                "ids": [0, 1],
                "goals": [[5.0, 5.0, 5.0], [10.0, 10.0, 10.0]]
            },
            "info": {
                "step": i,
                "collisions": []
            }
        })
    
    log_data = {
        "metadata": {"test": "data", "num_drones": 2},
        "frames": frames
    }
    
    if include_obstacles:
        log_data["metadata"]["obstacles"] = [
            {
                "type": "wall",
                "position": [5.0, 0.0, 5.0],
                "length": 10.0,
                "height": 10.0,
                "thickness": 0.5
            },
            {
                "type": "gate",
                "position": [5.0, 0.0, 3.0],
                "width": 3.0,
                "height": 6.0,
                "thickness": 0.5
            }
        ]
    
    with open(log_path, 'w') as f:
        json.dump(log_data, f)


def test_offline_visualizer_load_basic():
    """Test basic loading of JSON log file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test_log.json"
        create_test_log(log_path, num_frames=5)
        
        vis = OfflineVisualizer(log_path)
        vis.load()
        
        assert len(vis.frames) == 5
        assert vis.metadata["test"] == "data"
        assert vis.metadata["num_drones"] == 2


def test_offline_visualizer_load_with_obstacles():
    """Test loading log with obstacles in metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test_log.json"
        create_test_log(log_path, num_frames=3, include_obstacles=True)
        
        vis = OfflineVisualizer(log_path)
        vis.load()
        
        assert len(vis.frames) == 3
        assert len(vis.obstacles) == 2
        assert vis.obstacles[0]["type"] == "wall"
        assert vis.obstacles[1]["type"] == "gate"


def test_offline_visualizer_load_obstacles_from_metadata_environment():
    """Test loading obstacles from metadata.environment."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test_log.json"
        create_test_log(log_path, num_frames=3, include_obstacles=False)

        with open(log_path, "r") as f:
            data = json.load(f)
        data["metadata"]["environment"] = {
            "obstacles": [
                {
                    "type": "clutter",
                    "position": [1.0, 2.0, 3.0],
                    "length": 1.0,
                    "width": 1.0,
                    "height": 2.0,
                }
            ]
        }
        with open(log_path, "w") as f:
            json.dump(data, f)

        vis = OfflineVisualizer(log_path)
        vis.load()
        
        assert len(vis.obstacles) == 1
        assert vis.obstacles[0]["type"] == "clutter"


def test_offline_visualizer_missing_file():
    """Test that missing file raises appropriate error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "nonexistent.json"
        
        vis = OfflineVisualizer(log_path)
        
        with pytest.raises(FileNotFoundError):
            vis.load()


def test_offline_visualizer_empty_frames():
    """Test that log with no frames raises appropriate error."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "empty.json"
        
        with open(log_path, 'w') as f:
            json.dump({"metadata": {}, "frames": []}, f)
        
        vis = OfflineVisualizer(log_path)
        
        with pytest.raises(ValueError, match="No frames found"):
            vis.load()


def test_offline_visualizer_custom_playback_speed():
    """Test custom playback speed setting."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test_log.json"
        create_test_log(log_path, num_frames=3)
        
        vis = OfflineVisualizer(log_path, playback_speed=500.0)
        vis.load()
        
        assert vis.playback_speed == 500.0
        assert len(vis.frames) == 3


def test_offline_visualizer_frame_structure():
    """Test that frame structure is correctly parsed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test_log.json"
        create_test_log(log_path, num_frames=2)
        
        vis = OfflineVisualizer(log_path)
        vis.load()
        
        # Check first frame
        first_frame = vis.frames[0]
        assert "state" in first_frame
        assert "info" in first_frame
        
        state = first_frame["state"]
        assert "t" in state
        assert "pos" in state
        assert "vel" in state
        assert "acc" in state
        assert "ids" in state
        assert "goals" in state
        
        # Check that positions are lists
        assert isinstance(state["pos"], list)
        assert len(state["pos"]) == 2  # Two drones
        assert len(state["pos"][0]) == 3  # 3D position


def test_offline_visualizer_multiple_obstacle_types():
    """Test rendering different obstacle types."""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "test_log.json"
        
        # Create log with various obstacle types
        log_data = {
            "metadata": {
                "obstacles": [
                    {"type": "wall", "position": [0, 0, 0], "length": 5, "height": 3, "thickness": 0.1},
                    {"type": "gate", "position": [1, 1, 1], "width": 2, "height": 2, "thickness": 0.1},
                    {"type": "clutter", "position": [2, 2, 2], "length": 1, "width": 1, "height": 1},
                    {"type": "rectangular_prism", "position": [3, 3, 3], "length": 2, "width": 2, "height": 2},
                    # Test with old format (posx, posy, posz)
                    {"type": "unknown", "posx": 4, "posy": 4, "posz": 4, "width": 1, "height": 1, "depth": 1}
                ]
            },
            "frames": [
                {
                    "state": {"t": 0.0, "pos": [[0, 0, 0]], "vel": [[0, 0, 0]], "acc": [[0, 0, 0]], "ids": [0], "goals": [[1, 1, 1]]},
                    "info": {}
                }
            ]
        }
        
        with open(log_path, 'w') as f:
            json.dump(log_data, f)
        
        vis = OfflineVisualizer(log_path)
        vis.load()
        
        assert len(vis.obstacles) == 5
        
        # The _render_single_obstacle method should handle all these types
        # We can't test actual rendering without PyVista, but we can verify the data is loaded
        assert any(obs["type"] == "wall" for obs in vis.obstacles)
        assert any(obs["type"] == "gate" for obs in vis.obstacles)
        assert any(obs["type"] == "clutter" for obs in vis.obstacles)
        assert any(obs["type"] == "rectangular_prism" for obs in vis.obstacles)
        assert any(obs["type"] == "unknown" for obs in vis.obstacles)
