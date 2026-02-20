#!/usr/bin/env python3
"""
Monitor training progress using TensorBoard.

Launches TensorBoard pointing to the experiment's tensorboard log directory.
Can monitor a specific experiment or automatically detect the latest one.

Usage:
    python scripts/monitor_training.py                    # monitor latest experiment
    python scripts/monitor_training.py my_experiment      # monitor specific experiment
    python scripts/monitor_training.py my_experiment --port 6007  # custom port
    python scripts/monitor_training.py --list             # list all experiments
"""

import argparse
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path


def find_latest_experiment(experiments_dir: Path) -> Path | None:
    """Find the most recently modified experiment directory."""
    if not experiments_dir.exists():
        return None
    
    experiments = [d for d in experiments_dir.iterdir() if d.is_dir()]
    if not experiments:
        return None
    
    # Sort by modification time, most recent first
    experiments.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return experiments[0]


def resolve_experiment(name: str | None) -> Path:
    """Resolve experiment name to a directory, searching experiments/ if needed."""
    experiments_dir = Path("experiments")
    
    if name is None:
        # Auto-detect latest
        latest = find_latest_experiment(experiments_dir)
        if latest is None:
            sys.exit("No experiments found in experiments/ directory")
        return latest
    
    p = Path(name)
    if p.exists():
        return p.resolve()
    
    candidate = experiments_dir / name
    if candidate.exists():
        return candidate
    
    # Fuzzy: look for any subdir containing the name
    if experiments_dir.is_dir():
        matches = sorted(experiments_dir.iterdir())
        exact = [d for d in matches if d.name == name and d.is_dir()]
        if exact:
            return exact[0]
        partial = [d for d in matches if name in d.name and d.is_dir()]
        if len(partial) == 1:
            return partial[0]
        if len(partial) > 1:
            sys.exit(
                f"Ambiguous experiment name '{name}'. Matches:\n"
                + "\n".join(f"  {d.name}" for d in partial)
            )
    
    sys.exit(
        f"Experiment '{name}' not found "
        f"(checked ./{name} and experiments/{name})"
    )


def list_experiments():
    """List all available experiments."""
    experiments_dir = Path("experiments")
    if not experiments_dir.exists():
        print("No experiments/ directory found")
        return
    
    experiments = sorted(
        [d for d in experiments_dir.iterdir() if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )
    
    if not experiments:
        print("No experiments found")
        return
    
    print("Available experiments:")
    print("=" * 60)
    for exp in experiments:
        tensorboard_dir = exp / "tensorboard"
        has_logs = tensorboard_dir.exists() and any(tensorboard_dir.iterdir())
        status = "✓" if has_logs else "○"
        print(f"  {status} {exp.name}")
        if has_logs:
            print(f"      {tensorboard_dir}")


def can_use_programmatic_tensorboard():
    """Check if we can use the programmatic TensorBoard API."""
    try:
        # TensorBoard imports pkg_resources at runtime; verify it exists.
        if importlib.util.find_spec("pkg_resources") is None:
            return False
        from tensorboard import program
        return True
    except ImportError:
        return False


def launch_tensorboard(log_dir: Path, port: int = 6006):
    """Launch TensorBoard pointing to the given log directory."""
    if not log_dir.exists():
        sys.exit(f"TensorBoard log directory does not exist: {log_dir}")
    
    if not any(log_dir.iterdir()):
        sys.exit(
            f"TensorBoard log directory is empty: {log_dir}\n"
            "Training may not have started yet, or logs haven't been written."
        )
    
    print(f"Launching TensorBoard...")
    print(f"  Log directory: {log_dir}")
    print(f"  Port:          {port}")
    print(f"  URL:           http://localhost:{port}")
    print("=" * 60)
    print("Press Ctrl+C to stop TensorBoard")
    print("=" * 60)
    
    # Prefer programmatic approach to ensure we use the same Python environment
    if can_use_programmatic_tensorboard():
        # Use programmatic approach (most reliable)
        try:
            from tensorboard import program
            tb = program.TensorBoard()
            tb.configure(argv=[None, "--logdir", str(log_dir), "--port", str(port)])
            url = tb.launch()
            print(f"TensorBoard started at {url}")
            # Keep running until interrupted
            import time
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nTensorBoard stopped")
        except Exception as e:
            sys.exit(
                f"Failed to launch TensorBoard: {e}\n"
                "Make sure tensorboard and setuptools are installed:\n"
                "  uv pip install tensorboard setuptools"
            )
    else:
        # Fallback: try to find and use executable
        python_dir = Path(sys.executable).parent
        tensorboard_exe = python_dir / "tensorboard"
        
        if not tensorboard_exe.exists():
            tensorboard_exe = shutil.which("tensorboard")
        
        if tensorboard_exe:
            try:
                subprocess.run(
                    [str(tensorboard_exe), "--logdir", str(log_dir), "--port", str(port)],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                error_msg = str(e)
                if "pkg_resources" in error_msg or "ModuleNotFoundError" in error_msg:
                    sys.exit(
                        f"TensorBoard executable failed: {error_msg}\n\n"
                        "Missing dependency: pkg_resources (from setuptools)\n"
                        "Install it with:\n"
                        "  uv pip install --reinstall \"setuptools<81\"\n"
                        "  or\n"
                        "  pip install --force-reinstall \"setuptools<81\""
                    )
                sys.exit(f"Failed to launch TensorBoard: {e}")
        else:
            sys.exit(
                "TensorBoard not found. Install it with:\n"
                "  uv pip install tensorboard \"setuptools<81\"\n"
                "  or\n"
                "  pip install tensorboard \"setuptools<81\""
            )


def main():
    parser = argparse.ArgumentParser(
        description="Monitor training progress with TensorBoard"
    )
    parser.add_argument(
        "experiment",
        type=str,
        nargs="?",
        default=None,
        help="Experiment name (in experiments/) or full path. "
        "If not provided, uses the latest experiment.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6006,
        help="TensorBoard port (default: 6006)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available experiments and exit",
    )
    args = parser.parse_args()
    
    if args.list:
        list_experiments()
        return
    
    exp_dir = resolve_experiment(args.experiment)
    tensorboard_dir = exp_dir / "tensorboard"
    
    print(f"Experiment: {exp_dir.name}")
    print(f"Directory:  {exp_dir}")
    
    launch_tensorboard(tensorboard_dir, args.port)


if __name__ == "__main__":
    main()
