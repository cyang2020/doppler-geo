import numpy as np
import soundfile as sf

import dopgeo
from dopgeo import audio, measure, peaks, physics


def test_scan_finds_the_pass(one_pass):
    spec = audio.spectrogram(one_pass, audio.DEFAULT_SR)
    evs = [e for e in measure.scan(spec, log=None) if e.confident]
    assert evs, "the pass should be found"
    e = min(evs, key=lambda e: abs(e.t_heard - 120.0))
    assert abs(e.t_heard - 120.0) < 5.0
    assert abs(e.f0 - 110.0) < 2.0


def test_fastest_fall_matches_physics(one_pass):
    spec = audio.spectrogram(one_pass, audio.DEFAULT_SR)
    e = min(measure.scan(spec, log=None), key=lambda e: abs(e.t_heard - 120.0))
    ap = peaks.audio_peak(e)
    assert ap.ok
    # straight pass: the fastest fall is heard at the closest approach, at about kappa v^2 / (c d)
    c, v, d = physics.speed_of_sound(20.0), 70.0, 1500.0
    from dopgeo.kin import kappa

    expected = kappa(d / v) * v * v / (c * d)
    assert abs(ap.t - 120.0) <= ap.tol_t
    assert abs(np.log(ap.rate / expected)) < np.log(1.2)


def test_detect_passes_api(one_pass, tmp_path):
    p = tmp_path / "pass.wav"
    sf.write(p, one_pass, audio.DEFAULT_SR)
    ps = dopgeo.detect_passes(p)
    best = min(ps, key=lambda x: abs(x.time_s - 120.0))
    assert best.clear and best.straight
    assert best.fastest_fall_s is not None
