"""
Example script demonstrating the OfflineVisualizer for JSON simulation logs.

This script shows how to load and visualize simulation logs saved by
CoreSimulator.save_run() with step-by-step playback.

Usage:
    python examples/visualize_json_logs.py [path_to_log.json]

If no path is provided, it will use the demo_simulation_output.json file.
"""

from pathlib import Path
import sys

from flockrl_sim.visualization import OfflineVisualizer


def main():
    # Get log path from command line or use default
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        # Default to demo output in project root
        log_path = Path(__file__).resolve().parents[1] / "demo_simulation_output.json"
    
    if not log_path.exists():
        print(f"Error: Log file not found at {log_path}")
        print("Usage: python examples/visualize_json_logs.py [path_to_log.json]")
        return 1
    
    print(f"Loading simulation log from: {log_path}")
    
    # Create visualizer with custom playback speed (milliseconds between frames)
    vis = OfflineVisualizer(log_path, render_mode="plotly", playback_speed=250)
    
    # Load the data
    vis.load()
    
    print(f"Loaded {len(vis.frames)} frames")
    print(f"Found {len(vis.obstacles)} obstacles")
    
    if vis.metadata:
        print(f"Metadata: {vis.metadata}")
    
    vis.render()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
