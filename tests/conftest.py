import numpy as np
import pytest

from dopgeo import physics


def straight_pass_audio(sr=2000, duration=240.0, t_heard=120.0, f0=110.0, v=70.0, d=1500.0, amp=0.3, noise=0.05,
                        seed=0):
    """A mono recording of one aircraft tone passing in a straight line (exact Doppler shift, heard time),
    in white noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr
    f = physics.straight_line(t, t_heard, f0, v, d, physics.speed_of_sound(20.0))
    phase = 2 * np.pi * np.cumsum(f) / sr
    r = np.sqrt(d ** 2 + (v * (t - t_heard)) ** 2)
    a = amp * np.clip(d / r, 0.05, 1.0)
    return (a * np.sin(phase) + rng.normal(0, noise, len(t))).astype(np.float32)


@pytest.fixture
def one_pass():
    return straight_pass_audio()
