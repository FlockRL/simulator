from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple
import random
from math import hypot
from flockrl_sim.environment.obstacles_types import Obstacle, Wall, Gate, RectangularPrism, Bounds
from flockrl_sim.environment.spec_models.environment import EnvironmentSpec
from flockrl_sim.environment.spec_models.obstacles import WallSpec, ClutterSpec, GateSpec
from flockrl_sim.environment.spec_models.random_values import resolve_scalar, resolve_vector, resolve_partial_vector
from flockrl_sim.environment.validation import check_overlap, validate_environment
import logging

logger = logging.getLogger(__name__)

MAX_PLACEMENT_ATTEMPTS = 50
SPAWN_CLEARANCE_METERS = 2.0


class EnvironmentValidationError(Exception):
    """Raised when environment validation fails with errors."""
    pass


def _instance_id(base_id: str, index: int, total: int) -> str:
    """Generate instance ID with suffix if total > 1."""
    return f"{base_id}_{index}" if total > 1 else base_id


@dataclass
class Environment:
    bounds: Bounds
    obstacles: List[Obstacle]
    seed: Optional[int]

    def set_bounds(self, bounds: Bounds) -> None:
        self.bounds = bounds
        logger.debug(f"Environment bounds set to {self.bounds}")


    def add_obstacle(self, obstacle: Obstacle) -> None:
        self.obstacles.append(obstacle)
        logger.debug(f"Added obstacle: {obstacle}")

    def get_obstacle_by_id(self, obstacle_id: str) -> Optional[Obstacle]:
        obstacle = next((obs for obs in self.obstacles if obs.id == obstacle_id), None)
        if obstacle:
            logger.debug(f"Found obstacle: {obstacle}")
        return obstacle

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
    def __init__(self, config: Environment) -> None:
        self.config = config


    def add_random_obstacles(self, n: int = 5) -> "EnvironmentBuilder":
        if self.config.seed is not None:
            random.seed(self.config.seed)

        for i in range(n):
            x = random.uniform(self.config.bounds[0], self.config.bounds[1])
            y = random.uniform(self.config.bounds[2], self.config.bounds[3])
            z = random.uniform(self.config.bounds[4], self.config.bounds[5])
            obstacle = Obstacle(id=str(i), type="wall", position=(x, y, z), orientation=(0.0, 0.0, 0.0))
            self.config.add_obstacle(obstacle)
        return self

    @classmethod
    def from_spec(cls, spec: EnvironmentSpec) -> "EnvironmentBuilder":
        """
        Build environment from EnvironmentSpec (manual, random, or hybrid).

        Validates the environment and raises EnvironmentValidationError if invalid.
        """
        env = Environment(bounds=spec.bounds, obstacles=[], seed=spec.random_seed)
        if env.seed is not None:
            random.seed(env.seed)

        builder = cls(config=env)

        start_pos = resolve_vector(spec.start_position)
        goal_pos = resolve_vector(spec.goal_position)
        spawn_positions = [start_pos, goal_pos]

        for obs_spec in spec.obstacles:
            if isinstance(obs_spec, WallSpec):
                builder._process_wall_spec(obs_spec, spawn_positions)
            elif isinstance(obs_spec, ClutterSpec):
                builder._process_clutter_spec(obs_spec, spawn_positions)
            else:
                raise TypeError(f"Unsupported obstacle spec type: {type(obs_spec)}")

        validation_result = validate_environment(builder.config.obstacles, builder.config.bounds, start_pos, goal_pos) # Validate the environment

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
        spawn_positions: List[Tuple[float, float, float]],
    ) -> None:
        total = spec.count if spec.random else 1
        attempts = MAX_PLACEMENT_ATTEMPTS if spec.random else 1

        for index in range(total):
            wall_id = _instance_id(spec.id, index, total)
            placed = False

            for _ in range(attempts):
                position = resolve_vector(spec.position)
                orientation = resolve_vector(spec.orientation) if spec.orientation else (0.0, 0.0, 0.0)
                length = resolve_scalar(spec.length)
                height = resolve_scalar(spec.height)
                thickness = resolve_scalar(spec.thickness)

                # Build inline gates
                gate_instances = [
                    self._build_gate_instance(
                        gate_spec=gate_spec,
                        gate_id=f"{wall_id}_gate_{gate_idx}",
                        wall_position=position,
                        wall_orientation=orientation,
                        wall_thickness=thickness,
                    )
                    for gate_idx, gate_spec in enumerate(spec.gates)
                ]

                # Wall stores ID of first gate for backward compatibility
                first_gate_id = gate_instances[0].id if gate_instances else None

                wall = Wall(
                    id=wall_id,
                    type="wall",
                    position=position,
                    orientation=orientation,
                    length=length,
                    height=height,
                    thickness=thickness,
                    gate_id=first_gate_id,
                )

                if spec.random:
                    if not self._check_placement(wall, spawn_positions, {g.id for g in gate_instances} if gate_instances else None):
                        continue

                    # Check all gates for collisions
                    if not all(self._check_placement(g, spawn_positions, {wall.id}) for g in gate_instances):
                        continue

                # Add all gates first, then wall
                for gate_instance in gate_instances:
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
        gate_spec: GateSpec,
        gate_id: str,
        wall_position: Tuple[float, float, float],
        wall_orientation: Tuple[float, float, float],
        wall_thickness: float,
    ) -> Gate:
        position = resolve_partial_vector(gate_spec.position, wall_position)
        orientation = wall_orientation
        width = resolve_scalar(gate_spec.width)
        height = resolve_scalar(gate_spec.height)

        return Gate(
            id=gate_id,
            type="gate",
            position=position,
            orientation=orientation,
            width=width,
            height=height,
            thickness=wall_thickness,
        )

    def _process_clutter_spec(
        self,
        spec: ClutterSpec,
        spawn_positions: List[Tuple[float, float, float]],
    ) -> None:
        total = spec.count if spec.random else 1
        attempts = MAX_PLACEMENT_ATTEMPTS if spec.random else 1

        for index in range(total):
            clutter_id = _instance_id(spec.id, index, total)
            placed = False

            for _ in range(attempts):
                position = resolve_vector(spec.position)
                orientation = resolve_vector(spec.orientation) if spec.orientation else (0.0, 0.0, 0.0)
                length = resolve_scalar(spec.length)
                width = resolve_scalar(spec.width)
                height = resolve_scalar(spec.height)

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

                if spec.random and not self._check_placement(clutter, spawn_positions):
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
        return not spawn_positions or all(
            hypot(position[0] - spawn[0], position[1] - spawn[1]) >= SPAWN_CLEARANCE_METERS
            for spawn in spawn_positions
        )

    def _check_placement(
        self,
        obstacle: Obstacle,
        spawn_positions: List[Tuple[float, float, float]],
        ignore_ids: Optional[Set[str]] = None,
    ) -> bool:
        """Return True if obstacle placement is valid (clear of spawn and no collisions), False otherwise."""
        return (self._is_clear_of_spawn(obstacle.position, spawn_positions) and
                not self._collides_with_existing(obstacle, ignore_ids))

    def _collides_with_existing(
        self,
        candidate: Obstacle,
        ignore_ids: Optional[Set[str]] = None,
    ) -> bool:
        for existing in self.config.obstacles:
            if ignore_ids and existing.id in ignore_ids:
                continue
            if isinstance(candidate, Wall) and candidate.gate_id == existing.id:
                continue
            if isinstance(existing, Wall) and existing.gate_id == candidate.id:
                continue
            if check_overlap(candidate, existing):
                return True

        return False


    def _random_position_in_bounds(self, bounds: Bounds) -> Tuple[float, float, float]:
        return tuple(random.uniform(bounds[i], bounds[i+1]) for i in range(0, 6, 2))

    def build(self) -> Environment:
        return self.config
