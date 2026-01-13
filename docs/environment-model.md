# Environment Model

The environment module defines the spatial layout, obstacle types, and spec-driven construction for the simulator. It provides:

- `Environment`: bounds, obstacles, start and goal positions.
- `EnvironmentSpecLoader`: JSON spec loader with Pydantic validation.
- `EnvironmentBuilder`: resolves random values, instantiates obstacles, and validates placement.

## Obstacle types

`flockrl_sim/environment/obstacles_types.py` defines the obstacle dataclasses used by collision and perception:

- `Wall`: rectangular prism with optional linked gate IDs.
- `Gate`: rectangular prism used as a pass-through region.
- `RectangularPrism`: clutter geometry.

Each type implements `ray_intersect()` via OBB math in `flockrl_sim/geometry.py`.

## Spec-driven construction

`EnvironmentBuilder.from_spec()` consumes an `EnvironmentSpec` and resolves all random values with a seeded RNG. It performs these steps:

- Resolve start/goal positions.
- Instantiate wall and gate objects (gate positions can inherit from the parent wall).
- Instantiate clutter objects.
- Validate geometry, overlaps, gate embedding, and spawn locations.
- Enforce spawn clearance and re-sample when `random` placement is enabled.

Placement uses `max_placement_attempts` to bound retries for random obstacles. Gate embedding checks ensure gate volumes fit inside their parent walls.
