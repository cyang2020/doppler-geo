"""Shape table of one detected pass: for every (speed, closest distance) pair, how well a straight
constant-speed pass with those values fits the tracked tone line (best rest frequency and best
closest-approach time chosen freely), and which closest-approach time that fit puts it at.

The coarse search looks every (location, flight segment) pair up in this table: the flight's own
ADS-B speed and the geometric closest distance give a fit score and a predicted heard time, and
from that the recording start time. This uses what the curve really pins down (the joint shape),
not separate speed and distance ranges, which are poorly determined on their own.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from .measure import Event


@njit(cache=True, error_model="numpy")
def _factor(T, v, d, c):
    """Straight-pass Doppler factor at observer time T after the closest-approach sound."""
    T2 = T + d / c
    a = c * c - v * v
    disc = c ** 4 * T2 * T2 - a * (c * c * T2 * T2 - d * d)
    if disc < 0:
        disc = 0.0
    s = (c * c * T2 - np.sqrt(disc)) / a
    x = v * s
    r = np.sqrt(d * d + x * x)
    return c / (c + v * x / r)


@njit(parallel=True, cache=True, error_model="numpy")
def _table(lt, lf, vs, ds, taus, f0_lo, f0_hi, sigma, c):
    nv, nd, nt, n = len(vs), len(ds), len(taus), len(lt)
    Q = np.zeros((nv, nd))
    TC = np.zeros((nv, nd))
    F0 = np.zeros((nv, nd))
    SC = np.zeros((nv, nd, nt), np.float32)
    for k in prange(nv * nd):
        i = k // nd
        j = k % nd
        v, d = vs[i], ds[j]
        g = np.empty(n)
        best, bt, bf = -1.0, 0.0, 0.0
        for ti in range(nt):
            num = 0.0
            den = 0.0
            for m in range(n):
                g[m] = _factor(lt[m] - taus[ti], v, d, c)
                num += lf[m] * g[m]
                den += g[m] * g[m]
            f0 = min(max(num / den, f0_lo), f0_hi)
            for _ in range(2):  # robust re-weighting
                num = 0.0
                den = 0.0
                for m in range(n):
                    r = (lf[m] - f0 * g[m]) / (3 * sigma)
                    w = 1.0 / (1.0 + r * r)
                    num += w * lf[m] * g[m]
                    den += w * g[m] * g[m]
                f0 = min(max(num / den, f0_lo), f0_hi)
            tot = 0.0
            for m in range(n):
                r = (lf[m] - f0 * g[m]) / sigma
                tot += np.exp(-0.5 * r * r)
            tot /= n
            SC[i, j, ti] = tot
            if tot > best:
                best, bt, bf = tot, taus[ti], f0
        Q[i, j] = best
        TC[i, j] = bt
        F0[i, j] = bf
    return Q, TC, F0, SC


@dataclass
class ShapeTable:
    vs: np.ndarray  # speed grid (m/s), uniform
    lds: np.ndarray  # log10 distance grid, uniform
    Q: np.ndarray  # fit score in [0, 1], relative to the best cell (best = 1)
    TC: np.ndarray  # closest-approach heard time (audio s) of the best fit
    TLO: np.ndarray  # earliest / latest closest-approach time whose fit is still accepted
    THI: np.ndarray
    q_raw_max: float
    accept: float  # Q threshold used by the search

    def d_range(self):
        """Overall distance span (m) with any accepted cell."""
        ok = (self.Q >= self.accept).any(axis=0)
        if not ok.any():
            return 0.0, 0.0
        j = np.flatnonzero(ok)
        step = self.lds[1] - self.lds[0]
        return 10 ** (self.lds[j[0]] - step), 10 ** (self.lds[j[-1]] + step)

    def v_range(self):
        ok = (self.Q >= self.accept).any(axis=1)
        if not ok.any():
            return 0.0, 0.0
        i = np.flatnonzero(ok)
        step = self.vs[1] - self.vs[0]
        return self.vs[i[0]] - step, self.vs[i[-1]] + step

    def lookup(self, v, d):
        """Bilinear Q and TC at (v, d) (vectorised, for checks)."""
        return _lookup_vec(self.vs, self.lds, self.Q, self.TC, np.asarray(v, float), np.asarray(d, float))

    def accepted_area(self):
        """Fraction of the (v, log d) grid accepted: lower = more selective pass."""
        return float((self.Q >= self.accept).mean())


@njit(cache=True, error_model="numpy")
def lookup(vs, lds, Q, TC, v, d):
    """Bilinear interpolation of (Q, TC) at speed v, distance d; Q=0 outside the grid."""
    if d <= 0:
        return 0.0, 0.0
    x = (v - vs[0]) / (vs[1] - vs[0])
    y = (np.log10(d) - lds[0]) / (lds[1] - lds[0])
    nv, nd = Q.shape
    if x < 0 or y < 0 or x > nv - 1 or y > nd - 1:
        return 0.0, 0.0
    i = min(int(x), nv - 2)
    j = min(int(y), nd - 2)
    fx = x - i
    fy = y - j
    q = (Q[i, j] * (1 - fx) * (1 - fy) + Q[i + 1, j] * fx * (1 - fy) + Q[i, j + 1] * (1 - fx) * fy
         + Q[i + 1, j + 1] * fx * fy)
    # closest-approach time: take it from the best of the four corners (it can jump between fits)
    tc = TC[i, j]
    bq = Q[i, j]
    if Q[i + 1, j] > bq:
        bq, tc = Q[i + 1, j], TC[i + 1, j]
    if Q[i, j + 1] > bq:
        bq, tc = Q[i, j + 1], TC[i, j + 1]
    if Q[i + 1, j + 1] > bq:
        bq, tc = Q[i + 1, j + 1], TC[i + 1, j + 1]
    return q, tc


@njit(cache=True, error_model="numpy")
def _lookup_vec(vs, lds, Q, TC, v, d):
    q = np.zeros(len(v))
    tc = np.zeros(len(v))
    for k in range(len(v)):
        q[k], tc[k] = lookup(vs, lds, Q, TC, v[k], d[k])
    return q, tc


def build(e: Event, c=343.0, v_lo=2.0, v_hi=200.0, dv=2.0, d_lo=40.0, d_hi=20000.0, n_d=80, dtau=1.0,
          sigma_hz=None, accept=0.6, line_step=1.5) -> ShapeTable:
    t = np.asarray(e.line_t, float)
    f = np.asarray(e.line_f, float)
    keep = np.r_[0, np.flatnonzero(np.diff(np.floor(t / line_step)) > 0) + 1]
    t, f = t[keep], f[keep]
    if sigma_hz is None:
        sigma_hz = max(1.5, 0.008 * e.f0)
    vs = np.arange(v_lo, v_hi + 1e-9, dv)
    lds = np.linspace(np.log10(d_lo), np.log10(d_hi), n_d)
    pad = 30.0
    taus = np.arange(t[0] - pad, t[-1] + pad + 1e-9, dtau)
    f0_lo, f0_hi = f.min() * 0.85, f.max() * 1.15
    Q, TC, _, SC = _table(t, f, vs, 10 ** lds, taus, f0_lo, f0_hi, sigma_hz, c)
    qmax = float(max(Q.max(), 1e-9))
    ok = SC >= accept * qmax
    any_ok = ok.any(axis=2)
    first = np.argmax(ok, axis=2)
    last = len(taus) - 1 - np.argmax(ok[:, :, ::-1], axis=2)
    TLO = np.where(any_ok, taus[first] - dtau, TC)
    THI = np.where(any_ok, taus[last] + dtau, TC)
    return ShapeTable(vs=vs, lds=lds, Q=Q / qmax, TC=TC, TLO=TLO, THI=THI, q_raw_max=qmax, accept=accept)


def blurred(S: ShapeTable, r_m: float) -> ShapeTable:
    """Table for a whole grid cell: a cell is accepted if any point within r_m of its centre is.
    Q = max over closest distances d +/- r_m; the time interval is the union over those accepted."""
    if r_m <= 0:
        return S
    d = 10 ** S.lds
    nv, nd = S.Q.shape
    Q = np.zeros_like(S.Q)
    TLO = S.TLO.copy()
    THI = S.THI.copy()
    ok = S.Q >= S.accept
    for j in range(nd):
        a = np.searchsorted(d, d[j] - r_m)
        b = np.searchsorted(d, d[j] + r_m, side="right")
        a = max(0, min(a, j))
        b = max(b, j + 1)
        Q[:, j] = S.Q[:, a:b].max(axis=1)
        okw = ok[:, a:b]
        lo = np.where(okw, S.TLO[:, a:b], np.inf).min(axis=1)
        hi = np.where(okw, S.THI[:, a:b], -np.inf).max(axis=1)
        has = okw.any(axis=1)
        TLO[has, j] = lo[has]
        THI[has, j] = hi[has]
    return ShapeTable(vs=S.vs, lds=S.lds, Q=Q, TC=S.TC, TLO=TLO, THI=THI, q_raw_max=S.q_raw_max, accept=S.accept)
