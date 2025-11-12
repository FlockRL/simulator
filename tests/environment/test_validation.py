"""Unit tests for environment validation."""

import pytest

from flockrl_sim.environment.obstacles_types import Wall, Gate, RectangularPrism
from flockrl_sim.environment.validation import (
    validate_geometry,
    validate_no_overlaps,
    validate_gate_embedding,
    validate_environment,
)


class TestGeometryValidation:
    """Test individual obstacle geometry validation."""

    def test_valid_wall(self):
        """Test validation of a valid wall."""
        wall = Wall(
            id="wall1",
            type="wall",
            position=(0.0, 0.0, 0.0),
            length=5.0,
            height=3.0,
            thickness=0.1
        )
        bounds = (-10.0, 10.0, -10.0, 10.0, 0.0, 10.0)

        result = validate_geometry(wall, bounds)
        assert result.is_valid()

    def test_wall_outside_bounds(self):
        """Test validation catches wall outside bounds."""
        wall = Wall(
            id="wall1",
            type="wall",
            position=(15.0, 0.0, 0.0),  # Outside bounds
            length=5.0,
            height=3.0,
            thickness=0.1
        )
        bounds = (-10.0, 10.0, -10.0, 10.0, 0.0, 10.0)

        result = validate_geometry(wall, bounds)
        assert not result.is_valid()
        assert any("outside bounds" in err for err in result.errors)

    def test_wall_negative_dimensions(self):
        """Test validation catches negative dimensions."""
        wall = Wall(
            id="wall1",
            type="wall",
            position=(0.0, 0.0, 0.0),
            length=-5.0,  # Negative length
            height=3.0,
            thickness=0.1
        )
        bounds = (-10.0, 10.0, -10.0, 10.0, 0.0, 10.0)

        result = validate_geometry(wall, bounds)
        assert not result.is_valid()
        assert any("non-positive length" in err for err in result.errors)

    def test_valid_gate(self):
        """Test validation of a valid gate."""
        gate = Gate(
            id="gate1",
            type="gate",
            position=(0.0, 0.0, 1.0),
            width=1.5,
            height=1.5,
            frame_thickness=0.05
        )
        bounds = (-10.0, 10.0, -10.0, 10.0, 0.0, 10.0)

        result = validate_geometry(gate, bounds)
        assert result.is_valid()

    def test_valid_clutter(self):
        """Test validation of a valid clutter object."""
        clutter = RectangularPrism(
            id="clutter1",
            type="clutter",
            position=(2.0, 2.0, 0.0),
            subtype="rectangular_prism",
            length=0.5,
            width=0.5,
            height=0.8
        )
        bounds = (-10.0, 10.0, -10.0, 10.0, 0.0, 10.0)

        result = validate_geometry(clutter, bounds)
        assert result.is_valid()


class TestOverlapValidation:
    """Test overlap detection between obstacles."""

    def test_no_overlap(self):
        """Test that well-separated obstacles don't overlap."""
        obstacles = [
            Wall(
                id="wall1",
                type="wall",
                position=(0.0, 0.0, 0.0),
                length=5.0,
                height=3.0,
                thickness=0.1
            ),
            RectangularPrism(
                id="clutter1",
                type="clutter",
                position=(10.0, 10.0, 0.0),
                subtype="rectangular_prism",
                length=0.5,
                width=0.5,
                height=0.8
            )
        ]

        result = validate_no_overlaps(obstacles)
        assert result.is_valid()

    def test_overlap_detected(self):
        """Test that overlapping obstacles are detected."""
        obstacles = [
            RectangularPrism(
                id="clutter1",
                type="clutter",
                position=(0.0, 0.0, 0.0),
                subtype="rectangular_prism",
                length=1.0,
                width=1.0,
                height=1.0
            ),
            RectangularPrism(
                id="clutter2",
                type="clutter",
                position=(0.2, 0.2, 0.2),  # Very close, likely overlapping
                subtype="rectangular_prism",
                length=1.0,
                width=1.0,
                height=1.0
            )
        ]

        result = validate_no_overlaps(obstacles)
        assert not result.is_valid()
        assert any("may overlap" in err for err in result.errors)

    def test_gate_wall_overlap_ignored(self):
        """Test that gate-wall overlaps are ignored."""
        gate = Gate(
            id="gate1",
            type="gate",
            position=(0.0, 0.0, 1.0),
            width=1.5,
            height=1.5,
            frame_thickness=0.05
        )
        wall = Wall(
            id="wall1",
            type="wall",
            position=(0.0, 0.0, 0.0),
            length=5.0,
            height=3.0,
            thickness=0.1,
            gate_id="gate1"
        )

        obstacles = [wall, gate]
        result = validate_no_overlaps(obstacles)

        # Should be valid because gate-wall overlap is allowed
        assert result.is_valid()

    def test_overlap_respects_orientation(self):
        """Ensure overlap detection accounts for obstacle orientation."""
        wall = Wall(
            id="wall1",
            type="wall",
            position=(-3.0, 0.0, 0.0),
            length=10.0,
            height=3.0,
            thickness=0.2,
            orientation=(0.0, 0.0, 1.5708),
        )
        clutter = RectangularPrism(
            id="clutter1",
            type="clutter",
            position=(-3.0, 4.0, 0.0),  # Along wall's length axis
            subtype="rectangular_prism",
            length=1.0,
            width=1.0,
            height=1.0,
        )
        obstacles = [wall, clutter]
        result = validate_no_overlaps(obstacles)
        assert not result.is_valid(), "Clutter placed along rotated wall should overlap"


class TestGateEmbeddingValidation:
    """Test gate embedding in walls."""

    def test_valid_gate_wall_association(self):
        """Test valid gate-wall association."""
        gate = Gate(
            id="gate1",
            type="gate",
            position=(0.0, 0.0, 1.0),
            width=1.5,
            height=1.5,
            frame_thickness=0.05
        )
        wall = Wall(
            id="wall1",
            type="wall",
            position=(0.0, 0.0, 0.0),
            length=5.0,
            height=3.0,
            thickness=0.1,
            gate_id="gate1"
        )

        obstacles = [wall, gate]
        result = validate_gate_embedding(obstacles)

        # Should pass basic checks (gate exists and is at wall position)
        assert result.is_valid()

    def test_missing_gate_reference(self):
        """Test that referencing non-existent gate is caught."""
        wall = Wall(
            id="wall1",
            type="wall",
            position=(0.0, 0.0, 0.0),
            length=5.0,
            height=3.0,
            thickness=0.1,
            gate_id="nonexistent_gate"
        )

        obstacles = [wall]
        result = validate_gate_embedding(obstacles)

        assert not result.is_valid()
        assert any("non-existent gate" in err for err in result.errors)

    def test_gate_far_from_wall(self):
        """Test warning when gate is far from wall."""
        gate = Gate(
            id="gate1",
            type="gate",
            position=(10.0, 10.0, 1.0),  # Far from wall
            width=1.5,
            height=1.5,
            frame_thickness=0.05
        )
        wall = Wall(
            id="wall1",
            type="wall",
            position=(0.0, 0.0, 0.0),
            length=5.0,
            height=3.0,
            thickness=0.1,
            gate_id="gate1"
        )

        obstacles = [wall, gate]
        result = validate_gate_embedding(obstacles)

        # Should have a warning about distance
        assert len(result.warnings) > 0
        assert any("far from" in warn for warn in result.warnings)


class TestFullEnvironmentValidation:
    """Test full environment validation."""

    def test_valid_environment(self):
        """Test validation of a valid environment."""
        obstacles = [
            Wall(
                id="wall1",
                type="wall",
                position=(0.0, 0.0, 0.0),
                length=8.0,
                height=3.0,
                thickness=0.1,
                gate_id="gate1"
            ),
            Gate(
                id="gate1",
                type="gate",
                position=(0.0, 0.0, 1.0),
                width=1.5,
                height=1.5,
                frame_thickness=0.05
            ),
            RectangularPrism(
                id="clutter1",
                type="clutter",
                position=(5.0, 5.0, 0.0),
                subtype="rectangular_prism",
                length=0.5,
                width=0.5,
                height=0.8
            )
        ]
        bounds = (-10.0, 10.0, -10.0, 10.0, 0.0, 10.0)
        start = (-8.0, 0.0, 1.0)
        goal = (8.0, 0.0, 1.0)

        result = validate_environment(obstacles, bounds, start, goal)
        # May have warnings but should not have errors
        assert result.is_valid()

    def test_invalid_environment_multiple_errors(self):
        """Test environment with multiple validation errors."""
        obstacles = [
            Wall(
                id="wall1",
                type="wall",
                position=(0.0, 0.0, 0.0),
                length=-5.0,  # Invalid: negative length
                height=3.0,
                thickness=0.1,
                gate_id="nonexistent_gate"  # Invalid: gate doesn't exist
            ),
            RectangularPrism(
                id="clutter1",
                type="clutter",
                position=(0.1, 0.1, 0.1),  # May overlap with wall
                subtype="rectangular_prism",
                length=1.0,
                width=1.0,
                height=1.0
            )
        ]
        bounds = (-10.0, 10.0, -10.0, 10.0, 0.0, 10.0)

        result = validate_environment(obstacles, bounds)
        assert not result.is_valid()
        assert len(result.errors) >= 2  # Multiple errors detected
