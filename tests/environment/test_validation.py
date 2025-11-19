from flockrl_sim.environment.obstacles_types import Wall, Gate, RectangularPrism, Bounds
from flockrl_sim.environment.validation import (
    validate_geometry,
    validate_no_overlaps,
    validate_gate_embedding,
    validate_environment,
)


# Helper functions that use default values to make obstacle creation less verbose
def helper_wall(
    id="wall1",
    position=(0.0, 0.0, 0.0),
    orientation=(0.0, 0.0, 0.0),
    length=5.0,
    height=3.0,
    thickness=0.1,
    gate_ids=(),
):
    return Wall(
        id=id,
        type="wall",
        position=position,
        orientation=orientation,
        length=length,
        height=height,
        thickness=thickness,
        gate_ids=gate_ids,
    )


def helper_gate(
    id="gate1",
    position=(0.0, 0.0, 0.0),
    orientation=(0.0, 0.0, 0.0),
    width=1.5,
    height=1.5,
    thickness=0.05,
):
    return Gate(
        id=id,
        type="gate",
        position=position,
        orientation=orientation,
        width=width,
        height=height,
        thickness=thickness,
    )


def helper_clutter(
    id="clutter1",
    position=(0.0, 0.0, 0.0),
    orientation=(0.0, 0.0, 0.0),
    length=0.5,
    width=0.5,
    height=0.8,
):
    return RectangularPrism(
        id=id,
        type="clutter",
        position=position,
        orientation=orientation,
        subtype="rectangular_prism",
        length=length,
        width=width,
        height=height,
    )


# Common test constants
BOUNDS: Bounds = (-10.0, 10.0, -10.0, 10.0, 0.0, 10.0)


class TestGeometryValidation:
    """Test individual obstacle geometry validation."""

    def test_wall_outside_bounds(self):
        """Test validation catches wall outside bounds."""
        result = validate_geometry(helper_wall(position=(15.0, 0.0, 1.5)), BOUNDS)
        assert not result.is_valid()
        assert any("outside" in err and "bounds" in err for err in result.errors)

    def test_wall_negative_dimensions(self):
        """Test validation catches negative dimensions."""
        result = validate_geometry(helper_wall(length=-5.0), BOUNDS)
        assert not result.is_valid()
        assert any("non-positive length" in err for err in result.errors)


class TestOverlapValidation:
    """Test overlap detection between obstacles."""

    def test_overlap_detected(self):
        """Test that overlapping obstacles are detected."""
        obstacles = [
            helper_clutter(
                id="clutter1",
                position=(0.0, 0.0, 0.0),
                length=1.0,
                width=1.0,
                height=1.0,
            ),
            helper_clutter(
                id="clutter2",
                position=(0.2, 0.2, 0.2),
                length=1.0,
                width=1.0,
                height=1.0,
            ),
        ]
        result = validate_no_overlaps(obstacles)
        assert not result.is_valid()
        assert any("may overlap" in err for err in result.errors)

    def test_gate_wall_overlap_ignored(self):
        """Test that gate-wall overlaps are ignored."""
        g = helper_gate()
        w = helper_wall(gate_ids=("gate1",))
        result = validate_no_overlaps([w, g])
        assert result.is_valid()  # Gate-wall overlap is allowed

    def test_all_linked_gates_ignore_overlap(self):
        """Ensure every linked gate is treated as part of the parent wall."""
        g1 = helper_gate(id="gate1")
        g2 = helper_gate(id="gate2", position=(1.5, 0.0, 0.0))
        w = helper_wall(gate_ids=("gate1", "gate2"))
        result = validate_no_overlaps([w, g1, g2])
        assert result.is_valid()


class TestGateEmbeddingValidation:
    """Test gate embedding in walls."""

    def test_valid_gate_wall_association(self):
        """Test valid gate-wall association."""
        g = helper_gate()
        w = helper_wall(gate_ids=("gate1",))
        result = validate_gate_embedding([w, g])
        assert result.is_valid()  # Gate exists and is at wall position

    def test_missing_gate_reference(self):
        """Test that referencing non-existent gate is caught."""
        result = validate_gate_embedding([helper_wall(gate_ids=("nonexistent_gate",))])
        assert not result.is_valid()
        assert any("non-existent gate" in err for err in result.errors)

    def test_multiple_gate_references_are_validated(self):
        """Test that every gate linked to a wall is embedded."""
        g1 = helper_gate(id="gate1")
        g2 = helper_gate(id="gate2", position=(1.5, 0.0, 0.0))
        w = helper_wall(gate_ids=("gate1", "gate2"))
        result = validate_gate_embedding([w, g1, g2])
        assert result.is_valid()

    def test_gate_vertical_offset_detected(self):
        """Gates placed above the wall should fail validation."""
        g = helper_gate(position=(0.0, 0.0, 1.5))
        w = helper_wall(height=2.0, gate_ids=("gate1",))
        result = validate_gate_embedding([w, g])
        assert not result.is_valid()
        assert any("vertical span" in err for err in result.errors)


class TestFullEnvironmentValidation:
    """Test full environment validation."""

    def test_valid_environment(self):
        """Test validation of a valid environment."""
        obstacles = [
            helper_wall(
                id="wall1", length=8.0, gate_ids=("gate1",), position=(0.0, 0.0, 1.5)
            ),
            helper_gate(id="gate1", position=(0.0, 0.0, 1.5)),
            helper_clutter(id="clutter1", position=(5.0, 5.0, 0.4)),
        ]
        result = validate_environment(
            obstacles, BOUNDS, (-8.0, 0.0, 1.0), (8.0, 0.0, 1.0)
        )
        assert result.is_valid()  # May have warnings but should not have errors

    def test_invalid_environment_multiple_errors(self):
        """Test environment with multiple validation errors."""
        obstacles = [
            helper_wall(
                length=-5.0, gate_ids=("nonexistent_gate",)
            ),  # Invalid: negative length, missing gate
            helper_clutter(
                position=(0.1, 0.1, 0.1), length=1.0, width=1.0, height=1.0
            ),  # May overlap
        ]
        result = validate_environment(obstacles, BOUNDS)
        assert not result.is_valid()
        assert len(result.errors) >= 2  # Multiple errors detected

    def test_environment_with_multi_gate_wall_is_valid(self):
        """Walls with multiple gates should validate when gates exist."""
        obstacles = [
            helper_wall(
                id="wall1",
                position=(0.0, 0.0, 1.5),
                gate_ids=("gate1", "gate2"),
            ),
            helper_gate(id="gate1", position=(0.0, 0.0, 1.5)),
            helper_gate(id="gate2", position=(1.5, 0.0, 1.5)),
        ]
        result = validate_environment(
            obstacles, BOUNDS, (-8.0, 0.0, 1.0), (8.0, 0.0, 1.0)
        )
        assert result.is_valid()
