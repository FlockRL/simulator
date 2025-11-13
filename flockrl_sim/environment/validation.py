"""Environment validation helpers for geometry, collisions, and gate embedding.

Note: Collision detection uses conservative AABB approximation (see check_overlap).
"""

from __future__ import annotations

from typing import List, Tuple, Optional
from itertools import combinations
import math

from flockrl_sim.environment.obstacles_types import Obstacle, Wall, Gate, RectangularPrism


class ValidationResult:
    """Result of environment validation."""

    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def is_valid(self) -> bool:
        """Check if validation passed (no errors)."""
        return len(self.errors) == 0

    def __str__(self) -> str:
        """String representation of validation results."""
        if self.is_valid():
            msg = "Validation passed"
            if self.warnings:
                msg += f" ({len(self.warnings)} warnings)"
            return msg

        parts = [f"Validation failed with {len(self.errors)} errors:"]
        parts.extend(f"  - {error}" for error in self.errors)

        if self.warnings:
            parts.append(f"\nWarnings ({len(self.warnings)}):")
            parts.extend(f"  - {warning}" for warning in self.warnings)

        return "\n".join(parts)


def _extend_result(target: ValidationResult, source: ValidationResult) -> None:
    """Append errors and warnings from source into target."""
    target.errors.extend(source.errors)
    target.warnings.extend(source.warnings)


def _validate_positive_dimensions(
    result: ValidationResult,
    obstacle_id: str,
    obstacle_type: str,
    dimensions: List[Tuple[str, float]]
) -> None:
    """Check all dimensions are positive, add errors if not."""
    for name, value in dimensions:
        if value <= 0:
            result.add_error(f"{obstacle_type} {obstacle_id} has non-positive {name}: {value}")


def validate_geometry(
    obstacle: Obstacle,
    bounds: Tuple[float, float, float, float, float, float]
) -> ValidationResult:
    """Validate obstacle has positive dimensions and is within bounds."""
    result = ValidationResult()
    x, y, z = obstacle.position

    # Check position is within bounds
    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    for axis, coord, lower, upper in (
        ("x", x, x_min, x_max),
        ("y", y, y_min, y_max),
        ("z", z, z_min, z_max),
    ):
        if not lower <= coord <= upper:
            result.add_error(
                f"Obstacle {obstacle.id} {axis} position {coord} outside bounds [{lower}, {upper}]"
            )

    # Check type-specific dimensions
    if isinstance(obstacle, Wall):
        _validate_positive_dimensions(
            result, obstacle.id, "Wall",
            [("length", obstacle.length), ("height", obstacle.height), ("thickness", obstacle.thickness)]
        )
    elif isinstance(obstacle, Gate):
        _validate_positive_dimensions(
            result, obstacle.id, "Gate",
            [("width", obstacle.width), ("height", obstacle.height), ("thickness", obstacle.thickness)]
        )
    elif isinstance(obstacle, RectangularPrism):
        _validate_positive_dimensions(
            result, obstacle.id, "Clutter",
            [("length", obstacle.length), ("width", obstacle.width), ("height", obstacle.height)]
        )

    return result


def _axis_aligned_size(
    obs: Obstacle,
    dim_x: float,
    dim_y: float,
    dim_z: float,
) -> Tuple[float, float, float]:
    """Compute axis-aligned bounding box size accounting for yaw rotation."""
    yaw = 0.0
    if (orient := getattr(obs, "orientation", None)) and len(orient) >= 3 and orient[2] is not None:
        yaw = orient[2]

    cos_yaw, sin_yaw = abs(math.cos(yaw)), abs(math.sin(yaw))
    return (dim_x * cos_yaw + dim_y * sin_yaw,
            dim_x * sin_yaw + dim_y * cos_yaw,
            dim_z)


def _get_dims(obs: Obstacle) -> Tuple[float, float, float]:
    """Return local axis dimensions (length, width, height) for an obstacle."""
    match obs:
        case Wall():
            return obs.length, obs.thickness, obs.height
        case Gate():
            return obs.width, obs.thickness, obs.height
        case RectangularPrism():
            return obs.length, obs.width, obs.height
        case _:
            raise ValueError(f"Unsupported obstacle type: {type(obs)}")

def check_overlap(obs1: Obstacle, obs2: Obstacle) -> bool:
    """Conservative AABB collision check. May produce false positives for rotated obstacles.

    Returns True if AABBs overlap, False if no collision (only considers yaw rotation).
    """
    # Get positions
    x1, y1, z1 = obs1.position
    x2, y2, z2 = obs2.position

    # Get approximate sizes (conservative estimate)
    dims1 = _get_dims(obs1)
    dims2 = _get_dims(obs2)

    size1 = _axis_aligned_size(obs1, *dims1)
    size2 = _axis_aligned_size(obs2, *dims2)

    # Check overlap using axis-aligned bounding box test
    # Overlap occurs if distance between centers is less than sum of half-sizes in all dimensions
    deltas = (abs(x2 - x1), abs(y2 - y1), abs(z2 - z1))
    half_sum_sizes = ((size1[i] + size2[i]) / 2 for i in range(3))

    return all(d < hs for d, hs in zip(deltas, half_sum_sizes))


def _point_inside_obstacle(point: Tuple[float, float, float], obs: Obstacle) -> bool:
    """Check if a point is inside an obstacle's AABB."""
    x, y, z = point
    ox, oy, oz = obs.position
    dims = _get_dims(obs)
    size = _axis_aligned_size(obs, *dims)

    half_size_x, half_size_y, half_size_z = size[0] / 2, size[1] / 2, size[2] / 2
    return (abs(x - ox) < half_size_x and
            abs(y - oy) < half_size_y and
            abs(z - oz) < half_size_z)


def _is_gate_wall_pair(obs1: Obstacle, obs2: Obstacle) -> bool:
    """Check if obstacles are a gate-wall pair (allowed to overlap)."""
    return ((isinstance(obs1, Wall) and obs1.gate_id == obs2.id) or
            (isinstance(obs2, Wall) and obs2.gate_id == obs1.id))


def validate_no_overlaps(obstacles: List[Obstacle]) -> ValidationResult:
    """Check obstacles don't overlap (excludes gate-wall pairs)."""
    result = ValidationResult()

    for obs1, obs2 in combinations(obstacles, 2):
        if not _is_gate_wall_pair(obs1, obs2) and check_overlap(obs1, obs2):
            result.add_error(f"Obstacles {obs1.id} and {obs2.id} may overlap")

    return result


def validate_gate_embedding(obstacles: List[Obstacle]) -> ValidationResult:
    """Check gates are properly referenced by walls and positioned nearby."""
    result = ValidationResult()
    obs_map = {obs.id: obs for obs in obstacles}

    for obs in obstacles:
        if not isinstance(obs, Wall) or obs.gate_id is None:
            continue

        if obs.gate_id not in obs_map:
            result.add_error(f"Wall {obs.id} references non-existent gate {obs.gate_id}")
        elif not isinstance(gate := obs_map[obs.gate_id], Gate):
            result.add_error(f"Wall {obs.id} references {obs.gate_id} which is not a gate")
        elif (distance := math.dist(obs.position, gate.position)) > obs.length:
            result.add_warning(
                f"Gate {gate.id} is far from its parent wall {obs.id} (distance: {distance:.2f}m)"
            )

    return result


def validate_spawn_positions(
    obstacles: List[Obstacle],
    start_position: Optional[Tuple[float, float, float]],
    goal_position: Optional[Tuple[float, float, float]]
) -> ValidationResult:
    """Check start and goal positions are not inside obstacles."""
    result = ValidationResult()

    for position, name in [(start_position, "Start"), (goal_position, "Goal")]:
        if position is None:
            continue

        for obs in obstacles:
            if isinstance(obs, Gate):
                continue  # Gates are passable
            if _point_inside_obstacle(position, obs):
                result.add_error(f"{name} position {position} is inside obstacle {obs.id}")

    return result


def validate_environment(
    obstacles: List[Obstacle],
    bounds: Tuple[float, float, float, float, float, float],
    start_position: Optional[Tuple[float, float, float]] = None,
    goal_position: Optional[Tuple[float, float, float]] = None
) -> ValidationResult:
    """Validate geometry, overlaps, and gate embedding for all obstacles."""
    combined = ValidationResult()

    for obs in obstacles:
        _extend_result(combined, validate_geometry(obs, bounds))

    for validator in (validate_no_overlaps, validate_gate_embedding):
        _extend_result(combined, validator(obstacles))

    _extend_result(combined, validate_spawn_positions(obstacles, start_position, goal_position))

    return combined
