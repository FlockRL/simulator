from __future__ import annotations
from typing import List, Tuple, Optional
from itertools import combinations
import math
from flockrl_sim.environment.obstacles_types import Obstacle, Wall, Gate, RectangularPrism


class ValidationResult:
    """Result of environment validation."""

    def __init__(self):
        self.errors = []
        self.warnings = []

    def add_error(self, message: str) -> None:
        self.errors.append(message)

    def add_warning(self, message: str) -> None:
        self.warnings.append(message)

    def is_valid(self) -> bool:
        return not self.errors

    def __str__(self) -> str:
        if self.is_valid():
            return f"Validation passed{f' ({len(self.warnings)} warnings)' if self.warnings else ''}"

        parts = [f"Validation failed with {len(self.errors)} errors:"]
        parts.extend(f"  - {error}" for error in self.errors)
        if self.warnings:
            parts.extend([f"\nWarnings ({len(self.warnings)}):",
                         *(f"  - {warning}" for warning in self.warnings)])
        return "\n".join(parts)


def _extend_result(target: ValidationResult, source: ValidationResult) -> None:
    target.errors.extend(source.errors)
    target.warnings.extend(source.warnings)


def _validate_positive_dimensions(
    result: ValidationResult,
    obstacle_id: str,
    obstacle_type: str,
    dimensions: List[Tuple[str, float]]
) -> None:
    for name, value in dimensions:
        if value <= 0:
            result.add_error(f"{obstacle_type} {obstacle_id} has non-positive {name}: {value}")


def validate_geometry(
    obstacle: Obstacle,
    bounds: Tuple[float, float, float, float, float, float]
) -> ValidationResult:
    """Validate obstacle has positive dimensions and full extent is within bounds."""
    result = ValidationResult()
    x, y, z = obstacle.position
    x_min, x_max, y_min, y_max, z_min, z_max = bounds
    orientation = obstacle.orientation

    if abs(orientation[0]) > 1e-6 or abs(orientation[1]) > 1e-6:
        result.add_warning(f"Obstacle {obstacle.id} has roll/pitch rotations but validator currently only considers yaw when computing bounding boxes")

    dims = _get_dims(obstacle)
    aabb_size = _axis_aligned_size(obstacle, *dims)
    half_x, half_y, half_z = aabb_size[0] / 2, aabb_size[1] / 2, aabb_size[2] / 2

    if x - half_x < x_min or x + half_x > x_max:
        result.add_error(
            f"Obstacle {obstacle.id} extends outside x bounds "
            f"[{x - half_x:.2f}, {x + half_x:.2f}] not within [{x_min}, {x_max}]"
        )
    if y - half_y < y_min or y + half_y > y_max:
        result.add_error(
            f"Obstacle {obstacle.id} extends outside y bounds "
            f"[{y - half_y:.2f}, {y + half_y:.2f}] not within [{y_min}, {y_max}]"
        )
    if z - half_z < z_min or z + half_z > z_max:
        result.add_error(
            f"Obstacle {obstacle.id} extends outside z bounds "
            f"[{z - half_z:.2f}, {z + half_z:.2f}] not within [{z_min}, {z_max}]"
        )

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
    if (orient := getattr(obs, "orientation", None)) and len(orient) >= 3 and orient[2] != 0.0:
        yaw = orient[2]
    cos_yaw, sin_yaw = abs(math.cos(yaw)), abs(math.sin(yaw))
    return (dim_x * cos_yaw + dim_y * sin_yaw,
            dim_x * sin_yaw + dim_y * cos_yaw,
            dim_z)


def _get_dims(obs: Obstacle) -> Tuple[float, float, float]:
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
    x1, y1, z1 = obs1.position
    x2, y2, z2 = obs2.position

    size1 = _axis_aligned_size(obs1, *_get_dims(obs1))
    size2 = _axis_aligned_size(obs2, *_get_dims(obs2))

    # Overlap occurs if distance between centers is less than sum of half-sizes in all dimensions
    return all(abs(c2 - c1) < (s1 + s2) / 2
               for c1, c2, s1, s2 in zip((x1, y1, z1), (x2, y2, z2), size1, size2))


def _point_inside_obstacle(point: Tuple[float, float, float], obs: Obstacle) -> bool:
    size = _axis_aligned_size(obs, *_get_dims(obs))
    return all(abs(p - o) < s / 2 for p, o, s in zip(point, obs.position, size))


def _is_gate_wall_pair(obs1: Obstacle, obs2: Obstacle) -> bool:
    return ((isinstance(obs1, Wall) and obs2.id in obs1.linked_gate_ids()) or
            (isinstance(obs2, Wall) and obs1.id in obs2.linked_gate_ids()))


def validate_no_overlaps(obstacles: List[Obstacle]) -> ValidationResult:
    result = ValidationResult()

    for obs1, obs2 in combinations(obstacles, 2):
        if not _is_gate_wall_pair(obs1, obs2) and check_overlap(obs1, obs2):
            result.add_error(f"Obstacles {obs1.id} and {obs2.id} may overlap")

    return result


def validate_gate_embedding(obstacles: List[Obstacle]) -> ValidationResult:
    """Check gates are properly referenced by walls and fit within wall dimensions."""
    result = ValidationResult()
    obs_map = {obs.id: obs for obs in obstacles}

    for wall in (o for o in obstacles if isinstance(o, Wall)):
        for gate_id in wall.linked_gate_ids():
            if gate_id not in obs_map:
                result.add_error(f"Wall {wall.id} references non-existent gate {gate_id}")
                continue

            gate = obs_map[gate_id]
            if not isinstance(gate, Gate):
                result.add_error(f"Wall {wall.id} references {gate_id} which is not a gate")
                continue

            wall_half_length = wall.length / 2
            wall_half_height = wall.height / 2
            wall_half_thickness = wall.thickness / 2
            gate_half_width = gate.width / 2
            gate_half_height = gate.height / 2
            gate_half_thickness = gate.thickness / 2

            yaw = 0.0
            if (orient := getattr(wall, "orientation", None)) and len(orient) >= 3:
                yaw = orient[2]
            cos_yaw = math.cos(yaw)
            sin_yaw = math.sin(yaw)
            dx = gate.position[0] - wall.position[0]
            dy = gate.position[1] - wall.position[1]
            longitudinal_offset = abs(cos_yaw * dx + sin_yaw * dy)
            perpendicular_offset = abs(-sin_yaw * dx + cos_yaw * dy)

            if longitudinal_offset + gate_half_width > wall_half_length:
                result.add_error(
                    f"Gate {gate.id} extends beyond wall {wall.id} boundary "
                    f"(longitudinal offset: {longitudinal_offset:.2f}m + half-width: {gate_half_width:.2f}m "
                    f"> wall half-length: {wall_half_length:.2f}m)"
                )

            if perpendicular_offset + gate_half_thickness > wall_half_thickness:
                result.add_error(
                    f"Gate {gate.id} sits outside wall {wall.id} thickness plane "
                    f"(perpendicular offset: {perpendicular_offset:.2f}m + half-thickness: {gate_half_thickness:.2f}m "
                    f"> wall half-thickness: {wall_half_thickness:.2f}m)"
                )

            vertical_offset = abs(gate.position[2] - wall.position[2])
            if vertical_offset + gate_half_height > wall_half_height:
                result.add_error(
                    f"Gate {gate.id} sits outside wall {wall.id} vertical span "
                    f"(offset: {vertical_offset:.2f}m + half-height: {gate_half_height:.2f}m "
                    f"> wall half-height: {wall_half_height:.2f}m)"
                )

            if gate.height > wall.height:
                result.add_error(f"Gate {gate.id} height ({gate.height:.2f}m) exceeds parent wall {wall.id} height ({wall.height:.2f}m)")

    return result


def validate_spawn_positions(
    obstacles: List[Obstacle],
    start_position: Optional[Tuple[float, float, float]],
    goal_position: Optional[Tuple[float, float, float]],
    bounds: Tuple[float, float, float, float, float, float]
) -> ValidationResult:
    """Check start and goal positions are within bounds and not inside obstacles."""
    result = ValidationResult()
    x_min, x_max, y_min, y_max, z_min, z_max = bounds

    for position, name in [(start_position, "Start"), (goal_position, "Goal")]:
        if position is None:
            continue

        # Check if position is within bounds
        x, y, z = position

        if not (x_min <= x <= x_max):
            result.add_error(f"{name} position x-coordinate {x} is outside bounds [{x_min}, {x_max}]")
        if not (y_min <= y <= y_max):
            result.add_error(f"{name} position y-coordinate {y} is outside bounds [{y_min}, {y_max}]")
        if not (z_min <= z <= z_max):
            result.add_error(f"{name} position z-coordinate {z} is outside bounds [{z_min}, {z_max}]")

        # Check if position is inside any obstacle (excluding gates)
        for obs in (o for o in obstacles if not isinstance(o, Gate)):
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

    _extend_result(combined, validate_spawn_positions(obstacles, start_position, goal_position, bounds))

    return combined
