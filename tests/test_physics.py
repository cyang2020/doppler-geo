import numpy as np

from dopgeo import kin, physics


def test_speed_of_sound():
    assert abs(physics.speed_of_sound(20.0) - 343.2) < 0.5
    assert physics.speed_of_sound(0.0) < physics.speed_of_sound(30.0)


def test_straight_line_shape():
    c = 343.0
    t = np.array([0.0, 100.0, 200.0])
    f = physics.straight_line(t, 100.0, 120.0, 80.0, 1000.0, c)
    assert f[0] > f[1] > f[2]  # approaching -> higher, receding -> lower
    assert abs(f[1] - 120.0) < 1e-6  # rest frequency heard at the closest-approach time
    # far from the pass the shift tends to c / (c -+ v)
    far = physics.straight_line(np.array([-5000.0, 5000.0]), 0.0, 100.0, 80.0, 1000.0, c)
    assert abs(far[0] - 100.0 * c / (c - 80.0)) < 0.5
    assert abs(far[1] - 100.0 * c / (c + 80.0)) < 0.5


def test_kappa_limits():
    assert kin.kappa(100.0) == 1.0  # long passes are not flattened by the 5 s smoothing
    assert 0.0 < kin.kappa(2.0) < kin.kappa(10.0) < 1.0
    assert abs(kin.kappa(30.0) - kin._kappa_exact(30.0)) < 1e-3


def test_fall_rate_formula_at_closest_approach():
    # straight, constant velocity, listener abeam: rate = kappa * v^2 / (c R) (radial speed 0, no acceleration)
    c, v, R = 343.0, 80.0, 2000.0
    X = np.array([[0.0, R, 0.0]])
    V = np.array([[v, 0.0, 0.0]])
    A = np.zeros((1, 3))
    D, dist = kin._D(X, V, A, 0, 0.0, 0.0, 0.0, c)
    assert abs(dist - R) < 1e-9
    assert abs(D - kin.kappa(R / v) * v * v / (c * R)) < 1e-12


def test_turn_raises_fall_rate():
    # an aircraft accelerating away from the listener (turning away) makes the tone fall faster
    c, v, R = 343.0, 80.0, 2000.0
    X = np.array([[0.0, R, 0.0]])
    V = np.array([[v, 0.0, 0.0]])
    straight, _ = kin._D(X, V, np.zeros((1, 3)), 0, 0.0, 0.0, 0.0, c)
    away, _ = kin._D(X, V, np.array([[0.0, 2.0, 0.0]]), 0, 0.0, 0.0, 0.0, c)
    assert away > straight
