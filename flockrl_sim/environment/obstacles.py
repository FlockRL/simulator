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


def _instance_id(base_id: str, index: int, total: int) -> str:
    """Generate instance ID with suffix if total > 1."""
    return f"{base_id}_{index}" if total > 1 else base_id


@dataclass
class Environment:
    bounds: Bounds
    obstacles: List[Obstacle]
    start_position: Tuple[float, float, float]
    goal_position: Tuple[float, float, float]
    seed: int

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
        summary_lines = [
            f"Environment bounds: {self.bounds}",
            f"Seed: {self.seed}",
            f"Number of obstacles: {len(self.obstacles)}",
            f"Start position: {self.start_position}",
            f"Goal position: {self.goal_position}",
        ]
        return "\n".join(summary_lines)


class EnvironmentBuilder:
    def __init__(self, config: Environment, rng: random.Random) -> None:
        self.config = config
        self.rng = rng

    def add_random_obstacles(self, n: int = 5) -> "EnvironmentBuilder":
        for i in range(n):
            x = self.rng.uniform(self.config.bounds[0], self.config.bounds[1])
            y = self.rng.uniform(self.config.bounds[2], self.config.bounds[3])
            z = self.rng.uniform(self.config.bounds[4], self.config.bounds[5])
            obstacle = Obstacle(id=str(i), type="wall", position=(x, y, z), orientation=(0.0, 0.0, 0.0))
            self.config.add_obstacle(obstacle)
        return self

    @classmethod
    def from_spec(cls, spec: EnvironmentSpec) -> "EnvironmentBuilder":
        """Builds environment and validates it from EnvironmentSpec (manual, random, or hybrid)"""
        rng = random.Random(spec.random_seed)
        start_pos = resolve_vector(spec.start_position, rng)
        goal_pos = resolve_vector(spec.goal_position, rng)
        env = Environment(
            bounds=spec.bounds,
            obstacles=[],
            seed=spec.random_seed,
            start_position=start_pos,
            goal_position=goal_pos,
        )

        builder = cls(config=env, rng=rng)

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
            raise ValueError(f"Environment validation failed:\n{validation_result}")

        if validation_result.warnings:
            logger.warning(f"Environment validation warnings:\n{validation_result}")

        return builder

    def _process_wall_spec(self, spec: WallSpec, spawn_positions: List[Tuple[float, float, float]]) -> None:
        total = spec.count if spec.random else 1
        attempts = MAX_PLACEMENT_ATTEMPTS if spec.random else 1

        for index in range(total):
            wall_id = _instance_id(spec.id, index, total)
            placed = False

            for _ in range(attempts):
                position = resolve_vector(spec.position, self.rng)
                orientation = resolve_vector(spec.orientation, self.rng)
                length = resolve_scalar(spec.length, self.rng)
                height = resolve_scalar(spec.height, self.rng)
                thickness = resolve_scalar(spec.thickness, self.rng)

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

                wall = Wall(
                    id=wall_id,
                    type="wall",
                    position=position,
                    orientation=orientation,
                    length=length,
                    height=height,
                    thickness=thickness,
                    gate_ids=tuple(g.id for g in gate_instances),
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
                raise ValueError(
                    f"Unable to place wall '{wall_id}' without collisions after "
                    f"{MAX_PLACEMENT_ATTEMPTS} attempts. The environment may be too constrained "
                    f"(small bounds, large obstacles, or too many obstacles). "
                    f"Try: reducing obstacle count, increasing bounds, or decreasing obstacle size."
                )

    def _build_gate_instance(
        self,
        gate_spec: GateSpec,
        gate_id: str,
        wall_position: Tuple[float, float, float],
        wall_orientation: Tuple[float, float, float],
        wall_thickness: float,
    ) -> Gate:
        position = resolve_partial_vector(gate_spec.position, wall_position, self.rng)
        orientation = wall_orientation
        width = resolve_scalar(gate_spec.width, self.rng)
        height = resolve_scalar(gate_spec.height, self.rng)

        return Gate(
            id=gate_id,
            type="gate",
            position=position,
            orientation=orientation,
            width=width,
            height=height,
            thickness=wall_thickness,
        )

    def _process_clutter_spec(self, spec: ClutterSpec, spawn_positions: List[Tuple[float, float, float]]) -> None:
        total = spec.count if spec.random else 1
        attempts = MAX_PLACEMENT_ATTEMPTS if spec.random else 1

        for index in range(total):
            clutter_id = _instance_id(spec.id, index, total)
            placed = False

            for _ in range(attempts):
                position = resolve_vector(spec.position, self.rng)
                orientation = resolve_vector(spec.orientation, self.rng)
                length = resolve_scalar(spec.length, self.rng)
                width = resolve_scalar(spec.width, self.rng)
                height = resolve_scalar(spec.height, self.rng)

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
                raise ValueError(
                    f"Unable to place clutter '{clutter_id}' without collisions after "
                    f"{MAX_PLACEMENT_ATTEMPTS} attempts. The environment may be too constrained "
                    f"(small bounds, large obstacles, or too many obstacles). "
                    f"Try: reducing obstacle count, increasing bounds, or decreasing obstacle size."
                )

    def _is_clear_of_spawn(self, position, spawn_positions) -> bool:
        return not spawn_positions or all(
            hypot(position[0] - spawn[0], position[1] - spawn[1]) >= SPAWN_CLEARANCE_METERS
            for spawn in spawn_positions
        )

    def _check_placement(self, obstacle: Obstacle, spawn_positions: List[Tuple[float, float, float]], ignore_ids: Optional[Set[str]] = None) -> bool:
        """Return True if obstacle placement is valid (clear of spawn and no collisions), False otherwise."""
        return (self._is_clear_of_spawn(obstacle.position, spawn_positions) and
                not self._collides_with_existing(obstacle, ignore_ids))

    def _collides_with_existing(self, candidate: Obstacle, ignore_ids: Optional[Set[str]] = None) -> bool:
        """Return True if candidate collides with any existing obstacle, False otherwise."""
        for existing in self.config.obstacles:
            if ignore_ids and existing.id in ignore_ids:
                continue
            if isinstance(candidate, Wall) and existing.id in candidate.linked_gate_ids():
                continue
            if isinstance(existing, Wall) and candidate.id in existing.linked_gate_ids():
                continue
            if check_overlap(candidate, existing):
                return True

        return False

    def build(self) -> Environment:
        return self.config
