# Environment Specs

JSON files in this folder describe built-in environments used by the simulator.
Each file plugs directly into `EnvironmentSpecLoader`.

## Schema Snapshot

Top-level keys:

```json
{
  "name": "example",
  "bounds": [x_min, x_max, y_min, y_max, z_min, z_max],
  "random_seed": 123,
  "obstacles": [],
  "start_position": [x, y, z],
  "goal_position": [x, y, z]
}
```

Obstacles support `wall`, `gate`, and `clutter` types. Scalars accept either a
number, `{"uniform": [min, max]}`, or `{"discrete": [a, b, c]}`. Set
`"random": true` and `"count": N` to sample multiple instances deterministically
when a `random_seed` is provided.

Spawn zones require `start_position` and `goal_position`, which can be exact
coordinates or use the same random scalar system as obstacles (e.g.,
`[{"uniform": [-4.0, 4.0]}, {"uniform": [-4.0, 4.0]}, 0.0]`).

## Loading

```python
from flockrl_sim.environment import EnvironmentSpecLoader, EnvironmentBuilder

loader = EnvironmentSpecLoader()
spec = loader.load("simple")            # preset name
# spec = loader.load("/path/to/file")   # custom JSON
env = EnvironmentBuilder.from_spec(spec).build()
```

Create your own preset by copying one of these JSON files, editing obstacles and
spawn zones, and loading it with `load_from_path()` or the smart `load()`
method.

## Validation

All specs are validated on load:
- Bounds must be valid (min < max)
- Obstacle IDs must be unique
- Wall gate_id references must exist
- Dimensions must be positive
- Ranges must have min < max

Additional validation is performed during environment building:
- Obstacles must be within bounds
- No overlaps between obstacles
- Gates must be embedded within walls
- Spawn positions must be reachable
