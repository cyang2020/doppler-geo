"""The fastest fall of a pass: when the tone falls fastest, and how fast (log frequency per second).

On a 1 s grid the tracked line's log frequency is averaged over 5 s, then differenced over 4 s; the
largest fall and its time are the feature. Only changes of log frequency count, so the aircraft's rest
frequency drops out. For a straight constant-speed pass the fastest fall is heard at the closest approach
with rate about v^2 / (c d); for climbing, turning or accelerating aircraft it is not, which is why the
search (kin.py) predicts it from each flight's actual velocity and acceleration instead of a pass model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from .measure import Event
from .score import line_grid

HALF = 2  # moving average over 2*HALF+1 s, difference over 2*HALF s
MIN_RATE = 1.0e-3  # weaker peaks are too flat to time (1/s, i.e. 0.1 % of the frequency per second)


@dataclass
class AudioPeak:
    ok: bool
    t: float = 0.0  # audio time of the fastest fall (s)
    rate: float = 0.0  # fall rate of log frequency at that time (1/s)
    tol_t: float = 0.0  # how far the true peak time may be from t (s)


@njit(cache=True, error_model="numpy")
def fall_rate(L, ok, D, Dok, H, step):
    """D[j] = (M[j-H] - M[j+H]) / (2 H step) with M the (2H+1)-sample mean of L; Dok where every input was seen.
    On the 1 s grid H = 2; on a 2 s grid H = 1 spans the same 4 s (for curves smooth on that scale)."""
    n = len(L)
    for j in range(n):
        D[j] = 0.0
        Dok[j] = False
    for j in range(2 * H, n - 2 * H):
        good = True
        for u in range(j - 2 * H, j + 2 * H + 1):
            if not ok[u]:
                good = False
                break
        if not good:
            continue
        a = 0.0
        b = 0.0
        for u in range(-H, H + 1):
            a += L[j - H + u]
            b += L[j + H + u]
        D[j] = (a - b) / ((2 * H + 1) * 2 * H * step)
        Dok[j] = True


@njit(cache=True, error_model="numpy")
def _refine(D, j):
    """Sub-sample position and height of the maximum at j (parabola through j-1, j, j+1)."""
    if j <= 0 or j >= len(D) - 1:
        return 0.0, D[j]
    a = D[j - 1]
    b = D[j]
    c = D[j + 1]
    den = a - 2 * b + c
    if den >= 0:
        return 0.0, b
    x = 0.5 * (a - c) / den
    x = min(max(x, -0.5), 0.5)
    return x, b - 0.25 * (a - c) * x


def audio_peak(e: Event, edge_frac=0.6, margin_s=8.0) -> AudioPeak:
    """The pass's fastest fall, if the line shows it clearly: inside the seen stretch with the fall
    slowing down on both sides (to <= edge_frac of the peak) and at least margin_s from the ends."""
    if len(e.line_t) < 10:
        return AudioPeak(False)
    la, lf, seen = line_grid(e, 1.0)
    L = np.log(np.maximum(lf, 1e-3))
    D = np.zeros(len(L))
    Dok = np.zeros(len(L), bool)
    fall_rate(L, seen, D, Dok, HALF, 1.0)
    if not Dok.any():
        return AudioPeak(False)
    Dm = np.where(Dok, D, -np.inf)
    j = int(np.argmax(Dm))
    peak = D[j]
    if peak < MIN_RATE:
        return AudioPeak(False)
    # the seen stretch around j
    a = j
    while a > 0 and Dok[a - 1]:
        a -= 1
    b = j
    while b < len(D) - 1 and Dok[b + 1]:
        b += 1
    if j - a < margin_s or b - j < margin_s:
        return AudioPeak(False)
    if D[a:j + 1].min() > edge_frac * peak or D[j:b + 1].min() > edge_frac * peak:
        return AudioPeak(False)
    # full width at half maximum: broad peaks are timed less precisely
    h0 = j
    while h0 > a and D[h0 - 1] >= 0.5 * peak:
        h0 -= 1
    h1 = j
    while h1 < b and D[h1 + 1] >= 0.5 * peak:
        h1 += 1
    x, top = _refine(D, j)
    return AudioPeak(True, float(la[j] + x), float(top), float(2.0 + 0.15 * (h1 - h0)))
