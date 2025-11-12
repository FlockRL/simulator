"""Environment and obstacle data containers.

Coordinate with Collision team on obstacle geometry representation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import random
from math import hypot
import logging

logger = logging.getLogger(__name__)

from flockrl_sim.environment.obstacles_types import (
    Obstacle,
    Wall,
    Gate,
    RectangularPrism,
)
from flockrl_sim.environment.spec_models.environment import EnvironmentSpec
from flockrl_sim.environment.spec_models.obstacles import (
    WallSpec,
    GateSpec,
    ClutterSpec,
)
from flockrl_sim.environment.spec_models.random_values import (
    UniformRandomConfig,
    DiscreteRandomConfig,
)
from flockrl_sim.environment.validation import check_overlap

Bounds = Tuple[float, float, float, float, float, float]  # (x_min, x_max, y_min, y_max, z_min, z_max)
MAX_PLACEMENT_ATTEMPTS = 50
SPAWN_CLEARANCE_METERS = 2.0


class EnvironmentValidationError(Exception):
    """Raised when environment validation fails with errors."""
    pass

def _resolve_scalar(value) -> float:
    """Sample or coerce a scalar value."""
    if isinstance(value, UniformRandomConfig):
        low, high = value.uniform
        return random.uniform(low, high)
    if isinstance(value, DiscreteRandomConfig):
        return random.choice(value.discrete)
    return float(value)


def _resolve_vector(vector) -> Tuple[float, float, float]:
    """Resolve a 3D vector with potential random components."""
    return (
        _resolve_scalar(vector[0]),
        _resolve_scalar(vector[1]),
        _resolve_scalar(vector[2]),
    )


def _resolve_partial_vector(
    vector,
    fallback: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Resolve a partial vector using fallback components."""
    if vector is None:
        return fallback
    resolved = []
    for idx, component in enumerate(vector):
        if component is None:
            resolved.append(fallback[idx])
        else:
            resolved.append(_resolve_scalar(component))
    return tuple(resolved)


def _instance_suffix(index: int, total: int) -> str:
    """Return the suffix for the Nth instance."""
    if total <= 1:
        return ""
    return f"_{index}"


def _instance_id(base_id: str, suffix: str) -> str:
    """Compose an instance identifier with suffix."""
    return f"{base_id}{suffix}"



@dataclass
class Environment:
    bounds: Bounds = (-5.0, 5.0, -5.0, 5.0, -4.0, 4.0)
    obstacles: List[Obstacle] = field(default_factory=list)
    seed: Optional[int] = None

    def set_bounds(self, bounds: Bounds) -> None:
        self.bounds = bounds
        logger.debug(f"Environment bounds set to {self.bounds}")


    def add_obstacle(self, obstacle: Obstacle) -> None:
        self.obstacles.append(obstacle)
        logger.debug(f"Added obstacle: {obstacle}")

    def get_obstacle_by_id(self, obstacle_id: str) -> Optional[Obstacle]:
        for obs in self.obstacles:
            if obs.id == obstacle_id:
                logger.debug(f"Found obstacle: {obs}")
                return obs
        return None

    def summary(self) -> str:
        logger.debug("Generating environment summary")
        logger.debug(f"Bounds: {self.bounds}")
        logger.debug(f"Seed: {self.seed}")
        logger.debug(f"Number of obstacles: {len(self.obstacles)}")
        return (
            f"Environment bounds: {self.bounds}\n"
            f"Seed: {self.seed}\n"
            f"Number of obstacles: {len(self.obstacles)}"
        )


class EnvironmentBuilder:
    def __init__(self, config: Optional[Environment] = None, bounds: Optional[Bounds] = None) -> None:
        self.config = config or Environment()
        if bounds is not None:
            self.config.set_bounds(bounds)


    def add_random_obstacles(self, n: int = 5) -> "EnvironmentBuilder":
        if self.config.seed is not None:
            random.seed(self.config.seed)

        for i in range(n):
            x = random.uniform(self.config.bounds[0], self.config.bounds[1])
            y = random.uniform(self.config.bounds[2], self.config.bounds[3])
            z = random.uniform(self.config.bounds[4], self.config.bounds[5])
            obstacle = Obstacle(id=str(i), type="wall", position=(x, y, z))
            self.config.add_obstacle(obstacle)
        return self

    @classmethod
    def from_spec(cls, spec: EnvironmentSpec) -> "EnvironmentBuilder":
        """Build environment from EnvironmentSpec (manual, random, or hybrid).

        Validates obstacles and raises EnvironmentValidationError if invalid.
        """
        from flockrl_sim.environment.validation import validate_environment

        env = Environment(bounds=spec.bounds)
        if spec.random_seed is not None:
            env.seed = spec.random_seed
            random.seed(env.seed)

        builder = cls(config=env)

        spawn_zones = spec.spawn_zones
        start_pos = goal_pos = None
        spawn_positions: List[Tuple[float, float, float]] = []

        if spawn_zones:
            if spawn_zones.start_position is not None:
                start_pos = spawn_zones.start_position
            elif spawn_zones.start_zone_bounds is not None:
                start_pos = builder._random_position_in_bounds(spawn_zones.start_zone_bounds)

            if spawn_zones.goal_position is not None:
                goal_pos = spawn_zones.goal_position
            elif spawn_zones.goal_zone_bounds is not None:
                goal_pos = builder._random_position_in_bounds(spawn_zones.goal_zone_bounds)

            spawn_positions = [pos for pos in (start_pos, goal_pos) if pos]

        gate_templates: Dict[str, GateSpec] = {
            obs.id: obs for obs in spec.obstacles if isinstance(obs, GateSpec)
        }

        for obs_spec in spec.obstacles:
            if isinstance(obs_spec, WallSpec):
                builder._process_wall_spec(obs_spec, gate_templates, spawn_positions)
            elif isinstance(obs_spec, ClutterSpec):
                builder._process_clutter_spec(obs_spec, spawn_positions)
            elif isinstance(obs_spec, GateSpec):
                continue
            else:
                raise TypeError(f"Unsupported obstacle spec type: {type(obs_spec)}")

        # Run validation
        validation_result = validate_environment(
            builder.config.obstacles,
            builder.config.bounds,
            start_pos,
            goal_pos
        )

        if not validation_result.is_valid():
            raise EnvironmentValidationError(
                f"Environment validation failed:\n{validation_result}"
            )

        if validation_result.warnings:
            logger.warning(f"Environment validation warnings:\n{validation_result}")

        return builder

    def _process_wall_spec(
        self,
        spec: WallSpec,
        gate_templates: Dict[str, GateSpec],
        spawn_positions: List[Tuple[float, float, float]],
    ) -> None:
        """Instantiate walls (and optional gates) from a wall template."""
        total = spec.count if spec.random else 1
        attempts = MAX_PLACEMENT_ATTEMPTS if spec.random else 1

        for index in range(total):
            suffix = _instance_suffix(index, total)
            wall_id = _instance_id(spec.id, suffix)
            placed = False

            for _ in range(attempts):
                position = _resolve_vector(spec.position)
                orientation = (
                    _resolve_vector(spec.orientation)
                    if spec.orientation is not None
                    else (0.0, 0.0, 0.0)
                )
                length = _resolve_scalar(spec.length)
                height = _resolve_scalar(spec.height)
                thickness = _resolve_scalar(spec.thickness)

                gate_instance: Optional[Gate] = None
                if spec.gate_id is not None:
                    if spec.gate_id not in gate_templates:
                        raise ValueError(
                            f"Wall {spec.id} references missing gate template {spec.gate_id}"
                        )
                    gate_instance = self._build_gate_instance(
                        template=gate_templates[spec.gate_id],
                        gate_id=_instance_id(spec.gate_id, suffix),
                        wall_position=position,
                        wall_orientation=orientation,
                    )

                wall = Wall(
                    id=wall_id,
                    type="wall",
                    position=position,
                    orientation=orientation,
                    length=length,
                    height=height,
                    thickness=thickness,
                    gate_id=gate_instance.id if gate_instance else None,
                )

                if spec.random:
                    ignore_gate = {gate_instance.id} if gate_instance else None
                    if not self._is_clear_of_spawn(wall.position, spawn_positions):
                        continue
                    if self._collides_with_existing(wall, ignore_ids=ignore_gate):
                        continue

                    if gate_instance:
                        if not self._is_clear_of_spawn(
                            gate_instance.position, spawn_positions
                        ):
                            continue
                        if self._collides_with_existing(
                            gate_instance,
                            ignore_ids={wall.id},
                        ):
                            continue

                if gate_instance:
                    self.config.add_obstacle(gate_instance)
                self.config.add_obstacle(wall)
                placed = True
                break

            if not placed:
                logger.warning(
                    f"Unable to place wall '{wall_id}' "
                    f"without collisions after {MAX_PLACEMENT_ATTEMPTS} attempts"
                )

    def _build_gate_instance(
        self,
        template: GateSpec,
        gate_id: str,
        wall_position: Tuple[float, float, float],
        wall_orientation: Tuple[float, float, float],
    ) -> Gate:
        """Create a gate instance anchored to its parent wall."""
        position = _resolve_partial_vector(template.position, wall_position)
        orientation = _resolve_partial_vector(
            template.orientation,
            wall_orientation,
        )
        width = _resolve_scalar(template.width)
        height = _resolve_scalar(template.height)
        frame = _resolve_scalar(template.frame_thickness)

        return Gate(
            id=gate_id,
            type="gate",
            position=position,
            orientation=orientation,
            width=width,
            height=height,
            frame_thickness=frame,
        )

    def _process_clutter_spec(
        self,
        spec: ClutterSpec,
        spawn_positions: List[Tuple[float, float, float]],
    ) -> None:
        """Instantiate clutter obstacles from a template."""
        total = spec.count if spec.random else 1
        attempts = MAX_PLACEMENT_ATTEMPTS if spec.random else 1

        for index in range(total):
            suffix = _instance_suffix(index, total)
            clutter_id = _instance_id(spec.id, suffix)
            placed = False

            for _ in range(attempts):
                position = _resolve_vector(spec.position)
                orientation = (
                    _resolve_vector(spec.orientation)
                    if spec.orientation is not None
                    else (0.0, 0.0, 0.0)
                )
                length = _resolve_scalar(spec.length)
                width = _resolve_scalar(spec.width)
                height = _resolve_scalar(spec.height)

                clutter = RectangularPrism(
                    id=clutter_id,
                    type="clutter",
                    position=position,
                    orientation=orientation,
                    subtype=spec.subtype,
                    length=length,
                    width=width,
                    height=height,
                )

                if spec.random:
                    if not self._is_clear_of_spawn(clutter.position, spawn_positions):
                        continue
                    if self._collides_with_existing(clutter):
                        continue

                self.config.add_obstacle(clutter)
                placed = True
                break

            if not placed:
                logger.warning(
                    f"Unable to place clutter '{clutter_id}' "
                    f"without collisions after {MAX_PLACEMENT_ATTEMPTS} attempts"
                )

    def _is_clear_of_spawn(
        self,
        position: Tuple[float, float, float],
        spawn_positions: List[Tuple[float, float, float]],
    ) -> bool:
        """Ensure candidate position stays a safe distance from spawn points."""
        if not spawn_positions:
            return True

        x, y = position[0], position[1]
        for spawn in spawn_positions:
            if hypot(x - spawn[0], y - spawn[1]) < SPAWN_CLEARANCE_METERS:
                return False
        return True

    def _collides_with_existing(
        self,
        candidate: Obstacle,
        ignore_ids: Optional[Set[str]] = None,
    ) -> bool:
        """Return True if candidate overlaps any already placed obstacle."""
        ignore_ids = ignore_ids or set()

        for existing in self.config.obstacles:
            if existing.id in ignore_ids:
                continue
            if isinstance(candidate, Wall) and candidate.gate_id == existing.id:
                continue
            if isinstance(existing, Wall) and existing.gate_id == candidate.id:
                continue
            if check_overlap(candidate, existing):
                return True

        return False


    def _random_position_in_bounds(self, bounds: Bounds) -> Tuple[float, float, float]:
        """Generate random (x, y, z) position within given bounds."""
        x_min, x_max, y_min, y_max, z_min, z_max = bounds
        return (
            random.uniform(x_min, x_max),
            random.uniform(y_min, y_max),
            random.uniform(z_min, z_max)
        )

    def build(self) -> Environment:
        return self.config
