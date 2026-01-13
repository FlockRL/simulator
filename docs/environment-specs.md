# Environment Specs

This folder holds JSON environment presets consumed by `EnvironmentSpecLoader` and `EnvironmentBuilder`. Specs are validated using Pydantic models in `flockrl_sim/environment/spec_models`.

## Top-level schema

```json
{
  "name": "example",
  "description": "Short label",
  "bounds": [x_min, x_max, y_min, y_max, z_min, z_max],
  "random_seed": 123,
  "obstacles": [],
  "start_position": [x, y, z],
  "goal_position": [x, y, z]
}
```

## Obstacle schema

Each obstacle entry is one of:

- **Wall** (`type: "wall"`): `length`, `height`, `thickness`, optional `gates` list.
- **Clutter** (`type: "clutter"`): `subtype: "rectangular_prism"`, `length`, `width`, `height`.

Shared fields:

- `id`, `position`, `orientation`
- `random` (bool) and `count` (int). If any field uses a random value, `random` must be `true`.

### Gates

Gates are embedded inside walls and use a partial position. Any `null` component inherits the corresponding wall coordinate. Gate orientation always matches the parent wall orientation, and gate thickness is inherited from the wall thickness.

```json
{
  "position": [null, 0.0, null],
  "width": 3.0,
  "height": 4.0
}
```

## Random values

Scalar fields can be:

- A number
- `{ "uniform": [min, max] }`
- `{ "discrete": [a, b, c] }`

Vectors (`position`, `orientation`) are tuples of these scalars. Random obstacles with `count > 1` are expanded and assigned instance IDs using a numeric suffix.

## Validation

Validation happens at two stages:

- Pydantic model validation (bounds ordering, unique IDs, positive dimensions).
- Environment validation during build (bounds containment, overlap checks, gate embedding, spawn placement).
