"""
Greedy trajectory simplification for RL episode JSONs.

Usage:
    python scripts/simplify_trajectory.py <episode_json_path>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flockrl_sim.collision.system import CollisionSystem
from flockrl_sim.environment.obstacles import Environment
from flockrl_sim.environment.obstacles_types import Gate, RectangularPrism, Wall
from flockrl_sim.state import SwarmState

OBSTACLE_BUILDERS = {
    "gate": lambda obs, common: Gate(**common, width=obs["width"],
                                     height=obs["height"], thickness=obs["thickness"]),
    "wall": lambda obs, common: Wall(**common, length=obs["length"],
                                     height=obs["height"], thickness=obs["thickness"],
                                     gate_ids=tuple(obs.get("gate_ids", []))),
}


def build_obstacle(obs: dict):
    common = {"id": obs["id"], "type": obs["type"],
              "position": tuple(obs["position"]), "orientation": tuple(obs["orientation"])}
    if obs["type"] in OBSTACLE_BUILDERS:
        return OBSTACLE_BUILDERS[obs["type"]](obs, common)
    if obs["type"] == "clutter" and obs.get("subtype") == "rectangular_prism":
        return RectangularPrism(**common, subtype="rectangular_prism",
                                length=obs["length"], width=obs["width"],
                                height=obs["height"])
    return None


def build_environment(metadata: dict) -> Environment:
    env_data = metadata["environment"]
    obstacles = [o for obs in env_data.get("obstacles", [])
                 if (o := build_obstacle(obs)) is not None]
    return Environment(bounds=tuple(env_data["bounds"]), obstacles=obstacles,
                       start_position=tuple(env_data["start_position"]),
                       goal_position=tuple(env_data["goal_position"]), seed=None)


class SegmentChecker:
    def __init__(self, cs: CollisionSystem) -> None:
        self.cs = cs
        self.sample_step = cs.drone_radius / 2.0

    def is_clear(self, A: np.ndarray, B: np.ndarray) -> bool:
        seg = B - A
        length = float(np.linalg.norm(seg))
        if length < 1e-9:
            return self._point_clear(A)

        direction = seg / length
        n_steps = max(1, int(np.ceil(length / self.sample_step)))

        for k in range(n_steps + 1):
            t = min(k * self.sample_step, length)
            if not self._point_clear(A + t * direction):
                return False
        return True

    def _point_clear(self, p: np.ndarray) -> bool:
        r = self.cs.drone_radius
        bounds = self.cs.environment.bounds
        if (p[0] - r < bounds[0] or p[0] + r > bounds[1]
                or p[1] - r < bounds[2] or p[1] + r > bounds[3]
                or p[2] - r < bounds[4] or p[2] + r > bounds[5]):
            return False

        state = SwarmState.from_initial_positions(
            positions=p.reshape(1, 3), ids=np.array([0]), goals=p.reshape(1, 3))
        obstacles = self.cs.environment.obstacles
        if obstacles:
            if self.cs.check_wall_collision(state, obstacles):
                return False
            if self.cs.check_clutter_collision(state, obstacles):
                return False
        return True


def simplify_drone_path(positions: list[np.ndarray], checker: SegmentChecker) -> list[int]:
    n = len(positions)
    if n <= 2:
        return list(range(n))

    kept = [0]
    i = 0
    j = 1

    while j < n - 1:
        if checker.is_clear(positions[i], positions[j + 1]):
            j += 1
        else:
            kept.append(j)
            i = j
            j = i + 1

    kept.append(n - 1)
    return kept


def extract_positions_per_drone(frames: list[dict]) -> dict[int, tuple[list[np.ndarray], list[int]]]:
    positions: dict[int, list[np.ndarray]] = defaultdict(list)
    frame_indices: dict[int, list[int]] = defaultdict(list)

    for fi, frame in enumerate(frames):
        for did, pos in zip(frame["state"]["ids"], frame["state"]["pos"]):
            did = int(did)
            positions[did].append(np.array(pos, dtype=float))
            frame_indices[did].append(fi)

    return {did: (positions[did], frame_indices[did]) for did in positions}


def main() -> None:
    parser = argparse.ArgumentParser(description="Simplify an RL episode trajectory.")
    parser.add_argument("episode_json", type=Path)
    args = parser.parse_args()

    input_path = args.episode_json.resolve()
    with open(input_path) as f:
        episode = json.load(f)

    metadata, frames = episode["metadata"], episode["frames"]
    env = build_environment(metadata)
    drone_radius = float(metadata["config"]["collision"]["drone_radius"])
    checker = SegmentChecker(CollisionSystem(environment=env, drone_radius=drone_radius,
                                            restitution=1.0))

    positions_per_drone = extract_positions_per_drone(frames)
    kept_frame_indices: set[int] = set()

    for drone_id, (positions, frame_idx) in positions_per_drone.items():
        kept = simplify_drone_path(positions, checker)
        kept_frame_indices.update(frame_idx[k] for k in kept)
        reduction = 100.0 * (1.0 - len(kept) / len(positions))
        print(f"Drone {drone_id}: {len(positions)} → {len(kept)} waypoints ({reduction:.1f}% reduction)")

    simplified_frames = [frames[i] for i in sorted(kept_frame_indices)]
    output = {
        "metadata": {**metadata, "simplified": True,
                     "original_frame_count": len(frames),
                     "simplified_frame_count": len(simplified_frames)},
        "frames": simplified_frames,
    }

    output_path = input_path.with_stem(input_path.stem + "_simplified")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    total_reduction = 100.0 * (1 - len(simplified_frames) / len(frames))
    print(f"\nSaved to: {output_path}")
    print(f"Total: {len(frames)} → {len(simplified_frames)} frames ({total_reduction:.1f}% reduction)")


if __name__ == "__main__":
    main()
