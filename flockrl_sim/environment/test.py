import numpy as np
from math import isclose

# Import the obstacle classes from your package
from flockrl_sim.environment.obstacles_types import Gate, Wall, RectangularPrism


# -------------------------------------------------
# Utility
# -------------------------------------------------

def unit(v):
    v = np.array(v, dtype=float)
    return v / np.linalg.norm(v)

def assert_vec_close(a, b, tol=1e-6):
    a = np.array(a, float)
    b = np.array(b, float)
    assert np.allclose(a, b, atol=tol), f"Vectors differ: {a} vs {b}"


# -------------------------------------------------
# Tests
# -------------------------------------------------

def test_axis_aligned_hits():
    box = RectangularPrism(
        id="box1",
        type="box",
        position=(0,0,0),
        orientation=(0,0,0),
        length=2,
        width=2,
        height=2,
        subtype="box"
    )

    d = unit((0,0,-1))
    hit = box.ray_intersect(np.array([0,0,5], float), d, 100)
    assert hit is not None
    dist, p, n = hit
    assert isclose(dist, 4)
    assert_vec_close(p, [0,0,1])
    assert_vec_close(n, [0,0,1])

    d = unit((-1,0,0))
    hit = box.ray_intersect(np.array([5,0,0], float), d, 100)
    assert hit is not None
    dist, p, n = hit
    assert isclose(dist, 4)
    assert_vec_close(p, [1,0,0])
    assert_vec_close(n, [1,0,0])

    d = unit((0,-1,0))
    hit = box.ray_intersect(np.array([0,5,0], float), d, 100)
    assert hit is not None
    dist, p, n = hit
    assert isclose(dist, 4)
    assert_vec_close(p, [0,1,0])
    assert_vec_close(n, [0,1,0])

    print("Axis-aligned tests passed.")


def test_miss():
    box = RectangularPrism(
        id="box2",
        type="box",
        position=(0,0,0),
        orientation=(0,0,0),
        length=2,
        width=2,
        height=2,
        subtype="box"
    )

    d = unit((1,0,0))
    hit = box.ray_intersect(np.array([-5,5,0], float), d, 100)
    assert hit is None

    d = unit((0,1,0))
    hit = box.ray_intersect(np.array([3,0,0], float), d, 100)
    assert hit is None

    print("Miss tests passed.")


def test_rotated():
    box = RectangularPrism(
        id="box3",
        type="box",
        position=(0,0,0),
        orientation=(0,0,np.pi/2),
        length=2,
        width=2,
        height=2,
        subtype="box"
    )

    d = unit((-1,0,0))
    hit = box.ray_intersect(np.array([5,0,0], float), d, 100)
    assert hit is not None

    dist, p, n = hit
    assert isclose(dist, 4)
    # Correct expectation: world +X, not +Y
    assert_vec_close(n, [1,0,0])

    print("Rotated tests passed.")


def test_grazing():
    box = RectangularPrism(
        id="box4",
        type="box",
        position=(0,0,0),
        orientation=(0,0,0),
        length=2,
        width=2,
        height=2,
        subtype="box"
    )

    # Just inside +X face, so it should still intersect
    origin = np.array([1 - 1e-7, 0, 5], float)
    d = unit((0,0,-1))

    hit = box.ray_intersect(origin, d, 100)
    assert hit is not None

    print("Grazing tests passed.")



def test_wall_basic():
    wall = Wall(
        id="w1",
        type="wall",
        position=(0,0,0),
        orientation=(0,0,0),
        length=10,
        thickness=0.5,
        height=3
    )

    d = unit((-1,0,0))
    hit = wall.ray_intersect(np.array([20,0,0], float), d, 100)
    assert hit is not None
    dist, p, n = hit
    assert isclose(dist, 15)
    assert_vec_close(p, [5,0,0])
    assert_vec_close(n, [1,0,0])

    d = unit((0,0,-1))
    hit = wall.ray_intersect(np.array([0,0,10], float), d, 100)
    assert hit is not None
    dist, p, n = hit
    assert isclose(dist, 8.5)
    assert_vec_close(p, [0,0,1.5])
    assert_vec_close(n, [0,0,1])

    print("Wall basic tests passed.")


def test_wall_rotated():
    wall = Wall(
        id="w2",
        type="wall",
        position=(0,0,0),
        orientation=(0,0,np.pi/2),
        length=10,
        thickness=0.5,
        height=3
    )

    d = unit((0,-1,0))
    hit = wall.ray_intersect(np.array([0,20,0], float), d, 100)
    assert hit is not None
    dist, p, n = hit
    assert isclose(dist, 15)
    assert_vec_close(p, [0,5,0])
    assert_vec_close(n, [0,1,0])

    print("Wall rotated tests passed.")


def test_gate_basic():
    gate = Gate(
        id="g1",
        type="gate",
        position=(0,0,0),
        orientation=(0,0,0),
        width=4,
        height=5,
        frame_thickness=0.5
    )

    d = unit((-1,0,0))
    hit = gate.ray_intersect(np.array([10,0,0], float), d, 100)
    assert hit is not None
    dist, p, n = hit
    assert isclose(dist, 8)
    assert_vec_close(p, [2,0,0])
    assert_vec_close(n, [1,0,0])

    d = unit((0,0,-1))
    hit = gate.ray_intersect(np.array([0,0,10], float), d, 100)
    assert hit is not None
    dist, p, n = hit
    assert isclose(dist, 7.5)
    assert_vec_close(p, [0,0,2.5])
    assert_vec_close(n, [0,0,1])

    print("Gate basic tests passed.")


def test_gate_rotated():
    gate = Gate(
        id="g2",
        type="gate",
        position=(0,0,0),
        orientation=(0,np.pi/2,0),
        width=4,
        height=5,
        frame_thickness=0.5
    )

    d = unit((1,0,0))
    hit = gate.ray_intersect(np.array([-10,0,0], float), d, 100)
    assert hit is not None
    dist, p, n = hit
    assert isclose(dist, 7.5)
    assert_vec_close(p, [-2.5,0,0])

    print("Gate rotated tests passed.")


# -------------------------------------------------
# Runner
# -------------------------------------------------

def run_all_tests():
    print("Running tests...")
    test_axis_aligned_hits()
    test_miss()
    test_rotated()
    test_grazing()
    test_wall_basic()
    test_wall_rotated()
    test_gate_basic()
    test_gate_rotated()
    print("All tests passed.")


if __name__ == "__main__":
    run_all_tests()