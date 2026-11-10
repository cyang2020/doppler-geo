"""Kinematic search: (place, recording start) candidates from passes with a clear fastest fall.

Physics. A listener at p hears a flight's tone fall at the rate (log frequency per second)

    D(tau) = [ kappa (|v|^2 - rdot^2) / R + a . u ] * c / (c + rdot)^2

at emission time tau (R distance, u unit vector listener -> aircraft, rdot = v . u, v and a the
flight's velocity and acceleration smoothed over 9 s like the audio measurement, kappa the flattening
of sharp peaks by that smoothing). A pass with its fastest fall at audio time t_k and rate D_k came from
a flight whose D, seen from the true place, peaks at heard time T0 + t_k with height D_k. Turns and speed
changes are inside a, so manoeuvring aircraft need no special model.

Peak rows (depend only on the ADS-B data). For every flight track and every 1 km cell in reach, the
peaks of D are found by scanning the flight's 1 s samples around its closest approach to the cell centre.
Each peak is a row: its heard time and rate at the cell centre and how both change per metre east /
north (four nearby evaluations). Moving along a track delays the pass, so time and place are tied; with
the gradients a row is exact at any of 25 points 200 m apart inside the cell, at the cost of one 1 km
evaluation. Distant passes (> 4 km) are evaluated once per 2 x 2 cells and carried to each by the
gradients; close ones (< 1 cell) change too fast to linearise and get exact rows per point instead.

Join (depends on the audio). Each pass picks the rows whose rate fits it; a candidate is a place and
start time where many passes fit with one common start time. With m votes required out of Kp passes,
every candidate contains one of the Kp - m + 1 most precisely timed passes, so only their matches start a
search: per cell, each such match looks for the other passes in its time window, first over the whole
cell (exact for linear times), then at the 25 points. Cells are independent and run in parallel.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit, prange

from . import adsb, peaks
from .coarse import _seg_table, _GroundCfg
from .dem import DEM
from .score import TrajectoryBank
from .search import LocalProjection, Segments, _ground, _groups, _pad_bbox


@dataclass
class KinConfig:
    bbox: tuple
    t0_range: tuple
    audio_duration: float
    c: float = 343.0
    cell_m: float = 1000.0  # peak rows are made per cell of this size ...
    n_sub: int = 5  # ... and voted on at n_sub x n_sub points inside it
    block_m: float = 40000.0
    reach_m: float = 15000.0  # largest horizontal distance a pass is looked for at
    rate_tol: float = 1.3  # measured and predicted fall rates may differ by this factor
    min_votes: int = 3
    vote_frac: float = 0.6  # a candidate needs this share of the peak passes (and >= min_votes)
    keep: int = 3_000_000  # candidates passed on (ranked by votes, then checked with their matched flights)
    max_seg_gap_s: float = 60.0
    simplify_m: float = 25.0
    dem_zoom: int = 10
    obs_height: float = 1.5
    classes: tuple | None = None  # aircraft classes to consider (adsb.CLS_*); None = all
    probe: tuple | None = None  # (x, y, T0) of a known answer: log how it fares (debugging)
    log: callable = print


# --------------------------------------------------------------------------- flight samples

class Kinematics:
    """Every flight resampled to 1 s: local position (x, y in the search projection, z ellipsoidal height),
    velocity and acceleration (smoothed over +/- 4.5 s), flat arrays with a run index per sample (one run =
    one stretch of track without gaps)."""

    def __init__(self, pts: pd.DataFrame, proj, max_gap_s=30.0, half=4.5):
        p = pts.sort_values(["fid", "t"], kind="stable")
        h = adsb.fill_geometric_altitude(p)
        t = p.t.to_numpy(np.float64)
        fid = p.fid.to_numpy()
        lat, lon = p.lat.to_numpy(), p.lon.to_numpy()
        x, y = proj.xy(lat, lon)
        # the projection's local north (slightly rotated away from the centre)
        xn, yn = proj.xy(lat + 1e-4, lon)
        nx_, ny_ = xn - x, yn - y
        nn = np.hypot(nx_, ny_)
        nx_, ny_ = nx_ / nn, ny_ / nn
        gs = p.gs.to_numpy(np.float64)
        trk = np.radians(p.track.to_numpy(np.float64))
        vr = np.nan_to_num(p.vrate.to_numpy(np.float64))
        ve, vn = gs * np.sin(trk), gs * np.cos(trk)
        vx = ve * ny_ + vn * nx_
        vy = -ve * nx_ + vn * ny_
        T, F, X, V = _resample(t, fid.astype(np.int64), x, y, h.astype(np.float64), vx, vy, vr, max_gap_s)
        self.t, self.fid, self.X, self.V = T, F, X, V
        run = np.r_[0, np.cumsum((np.diff(T) != 1.0) | (F[1:] != F[:-1]))]
        self.run = run.astype(np.int64)
        self.A = _accel(V, self.run, int(round(half)))

    def index(self, fid, t):
        """First sample of flight fid at or after time t, if within 2 s (-1 if none)."""
        key = self.fid * 4.0e6 + (self.t - self._t0)
        q = np.asarray(fid) * 4.0e6 + (np.asarray(t, np.float64) - self._t0)
        j = np.minimum(np.searchsorted(key, q - 1e-6), len(key) - 1)
        ok = (self.fid[j] == np.asarray(fid)) & (self.t[j] - np.asarray(t, np.float64) <= 2.0) & (self.t[j] >= np.asarray(t) - 1e-6)
        return np.where(ok, j, -1).astype(np.int64)

    @property
    def _t0(self):
        return float(np.floor(self.t.min()))


@njit(cache=True, error_model="numpy")
def _resample(t, fid, x, y, h, vx, vy, vz, max_gap):
    """Linear resampling of each flight to whole seconds; stretches longer than max_gap without data break it."""
    n = len(t)
    good = np.isfinite(vx) & np.isfinite(vy) & np.isfinite(x) & np.isfinite(h)
    cnt = 0
    a = 0
    while a < n:
        b = a + 1
        while b < n and fid[b] == fid[a]:
            b += 1
        prev = -1
        for i in range(a, b):
            if not good[i]:
                continue
            if prev >= 0 and t[i] - t[prev] <= max_gap and t[i] > t[prev] + 0.05:
                cnt += int(np.floor(t[i]) - np.floor(t[prev]))
            prev = i
        a = b
    T = np.empty(cnt)
    F = np.empty(cnt, np.int64)
    X = np.empty((cnt, 3))
    V = np.empty((cnt, 3))
    m = 0
    a = 0
    while a < n:
        b = a + 1
        while b < n and fid[b] == fid[a]:
            b += 1
        prev = -1
        for i in range(a, b):
            if not good[i]:
                continue
            if prev >= 0 and t[i] - t[prev] <= max_gap and t[i] > t[prev] + 0.05:
                s0 = np.floor(t[prev]) + 1.0
                while s0 <= t[i] and m < cnt:
                    f = (s0 - t[prev]) / (t[i] - t[prev])
                    T[m] = s0
                    F[m] = fid[a]
                    X[m, 0] = x[prev] + f * (x[i] - x[prev])
                    X[m, 1] = y[prev] + f * (y[i] - y[prev])
                    X[m, 2] = h[prev] + f * (h[i] - h[prev])
                    V[m, 0] = vx[prev] + f * (vx[i] - vx[prev])
                    V[m, 1] = vy[prev] + f * (vy[i] - vy[prev])
                    V[m, 2] = vz[prev] + f * (vz[i] - vz[prev])
                    m += 1
                    s0 += 1.0
            prev = i
        a = b
    return T[:m], F[:m], X[:m], V[:m]


@njit(cache=True, error_model="numpy")
def _accel(V, run, k):
    n = len(V)
    A = np.zeros((n, 3))
    for j in range(n):
        a = j
        b = j
        while a > 0 and j - a < k and run[a - 1] == run[j]:
            a -= 1
        while b < n - 1 and b - j < k and run[b + 1] == run[j]:
            b += 1
        if b > a:
            for q in range(3):
                A[j, q] = (V[b, q] - V[a, q]) / (b - a)
    return A


# --------------------------------------------------------------------------- fall-rate peaks

@njit(cache=True, error_model="numpy")
def _kappa_exact(w):
    """Peak of the 5 s averaged, 4 s differenced fall rate of a straight pass relative to the true peak,
    for a closest approach lasting w = d / v seconds."""
    if w > 60.0:
        return 1.0
    m_lo = 0.0
    m_hi = 0.0
    for u in range(-2, 3):
        t = (u - 2.0) / w
        m_lo += t / np.sqrt(1.0 + t * t)
        t = (u + 2.0) / w
        m_hi += t / np.sqrt(1.0 + t * t)
    return (m_hi - m_lo) / 20.0 * w


def _kappa_table():
    w = np.arange(0.0, 60.01, 0.25)
    return np.array([_kappa_exact(max(x, 1e-3)) for x in w])


KAPPA = _kappa_table()


@njit(cache=True, error_model="numpy")
def kappa(w):
    """_kappa_exact from a 0.25 s table (1 beyond 60 s)."""
    if w >= 60.0:
        return 1.0
    x = w * 4.0
    i = int(x)
    f = x - i
    return KAPPA[i] * (1.0 - f) + KAPPA[i + 1] * f


@njit(cache=True, error_model="numpy")
def _D(X, V, A, j, px, py, pz, c):
    """Fall rate heard at (px, py, pz) from sample j, and the distance."""
    rx = X[j, 0] - px
    ry = X[j, 1] - py
    rz = X[j, 2] - pz
    R = np.sqrt(rx * rx + ry * ry + rz * rz)
    ux, uy, uz = rx / R, ry / R, rz / R
    vx, vy, vz = V[j, 0], V[j, 1], V[j, 2]
    v2 = vx * vx + vy * vy + vz * vz
    rd = vx * ux + vy * uy + vz * uz
    k = kappa(R / max(np.sqrt(v2), 1.0))
    return (k * (v2 - rd * rd) / R + A[j, 0] * ux + A[j, 1] * uy + A[j, 2] * uz) * c / ((c + rd) * (c + rd)), R


NPK = 3  # fall-rate peaks kept per flight and cell
SEP_S = 8.0  # peaks closer than this are one peak


@njit(cache=True, error_model="numpy")
def _maxima(X, V, A, run, ts, j0, j1, step, px, py, pz, c, cand, nmax):
    """Local maxima of D over samples j0..j1 (same run) every `step` (on the 1 s grid D is first averaged over
    3 samples, against jitter in the ADS-B velocities): cand[m] = (heard time, height, sample). Returns how many."""
    n = 0
    wa = 1.0 / 3.0 if step == 1 else 0.0
    wb = 1.0 - 2.0 * wa
    j = j0
    Da, Ra = _D(X, V, A, j, px, py, pz, c)
    Db, Rb = _D(X, V, A, j + step, px, py, pz, c)
    Dc, Rc = _D(X, V, A, j + 2 * step, px, py, pz, c)
    S0 = wa * (Da + Dc) + wb * Db  # centred on j + step
    Dd, Rd = _D(X, V, A, j + 3 * step, px, py, pz, c)
    S1 = wa * (Db + Dd) + wb * Dc  # centred on j + 2 step
    Da, Db, Dc = Db, Dc, Dd
    Rb_, Rc_ = Rc, Rd
    jj = j + 4 * step
    while jj <= j1:
        De, Re = _D(X, V, A, jj, px, py, pz, c)
        S2 = wa * (Db + De) + wb * Dc  # centred on jj - step
        jm = jj - 2 * step  # centre of S1
        if S1 >= S0 and S1 > S2 and S1 > 0 and n < nmax:
            den = S0 - 2 * S1 + S2
            x = 0.0
            top = S1
            if den < 0:
                x = min(max(0.5 * (S0 - S2) / den, -0.5), 0.5)
                top = S1 - 0.25 * (S0 - S2) * x
            cand[n, 0] = ts[jm] + x * step + Rb_ / c
            cand[n, 1] = top
            cand[n, 2] = jm
            n += 1
        S0, S1 = S1, S2
        Db, Dc = Dc, De
        Rb_, Rc_ = Rc_, Re
        jj += step
    return n


@njit(cache=True, error_model="numpy")
def _scan(X, V, A, run, ts, jc, half, step, px, py, pz, c, out, cand):
    """Peaks of D over the samples jc - half .. jc + half of jc's run: peaks closer than SEP_S merged (higher
    kept), then the NPK highest that reach 0.4 of the highest. out[p] = (heard time, height, sample)."""
    nr = len(run)
    rj = run[jc]
    j0 = max(jc - half, 0)
    while run[j0] != rj:
        j0 += 1
    j1 = min(jc + half, nr - 1)
    while run[j1] != rj:
        j1 -= 1
    if j1 - j0 < 5 * step:
        return 0
    n = _maxima(X, V, A, run, ts, j0, j1, step, px, py, pz, c, cand, len(cand))
    if n == 0:
        return 0
    # merge close peaks
    for a in range(n):
        if cand[a, 1] <= 0:
            continue
        for b in range(a + 1, n):
            if cand[b, 1] > 0 and abs(cand[b, 0] - cand[a, 0]) < SEP_S:
                if cand[b, 1] > cand[a, 1]:
                    cand[a, 1] = -1.0
                    break
                cand[b, 1] = -1.0
    top = 0.0
    for a in range(n):
        top = max(top, cand[a, 1])
    m = 0
    while m < NPK:
        bi = -1
        for a in range(n):
            if cand[a, 1] >= 0.4 * top and (bi < 0 or cand[a, 1] > cand[bi, 1]):
                bi = a
        if bi < 0:
            break
        out[m, 0] = cand[bi, 0]
        out[m, 1] = cand[bi, 1]
        out[m, 2] = cand[bi, 2]
        cand[bi, 1] = -1.0
        m += 1
    return m


@njit(cache=True, error_model="numpy")
def _near_peak(X, V, A, run, ts, js, px, py, pz, c, cand, win):
    """The peak (3-sample average, 1 s grid) nearest sample js within win s: (ok, heard time, height)."""
    nr = len(run)
    rj = run[js]
    j0 = max(js - win, 0)
    j1 = min(js + win, nr - 1)
    while run[j0] != rj:
        j0 += 1
    while run[j1] != rj:
        j1 -= 1
    if j1 - j0 < 6:
        return False, 0.0, 0.0
    n = _maxima(X, V, A, run, ts, j0, j1, 1, px, py, pz, c, cand, len(cand))
    bi = -1
    for a in range(n):
        if bi < 0 or abs(cand[a, 2] - js) < abs(cand[bi, 2] - js):
            bi = a
    if bi < 0:
        return False, 0.0, 0.0
    return True, cand[bi, 0], cand[bi, 1]


# --------------------------------------------------------------------------- peak rows (independent of the audio)

NF = 11  # row fields: cell, heard time (rel.), dT/dx, dT/dy, time residual, lnD, dlnD/dx, dlnD/dy, extra lnD tol,
#           sample, point (-1: the row covers the whole cell through its gradients; else only that sub-point)
NEAR_CELLS = 1.0  # closer than this many cells, peaks are evaluated at every sub-point instead of linearised


@njit(cache=True, error_model="numpy")
def _cell_peaks(i, ix, iy, x0, y0, nx, ny, g, ground, mask, SG, KB, X, V, A, run, ts, lDk, lt, reach, c, th_lo,
                th_hi, out, pos, cap, wout, cand, n_sub):
    """Fall-rate peaks of segment i's flight heard at cell (ix, iy): heard time and rate at the cell centre and
    how both change per metre. Written to out[pos:]; returns the new pos (> cap: overflow). With lDk non-empty
    only peaks within reach of some pass's rate (ln D_k +/- lt plus the cell's spread) are kept."""
    cx = x0 + (ix + 0.5) * g
    cy = y0 + (iy + 0.5) * g
    cz = ground[iy, ix]
    Ax, Ay, Az, At = SG[i, 0], SG[i, 1], SG[i, 2], SG[i, 3]
    dx, dy, dz = SG[i, 5] - Ax, SG[i, 6] - Ay, SG[i, 7] - Az
    L2 = dx * dx + dy * dy + dz * dz
    if L2 < 1e-6:
        return pos
    s = ((cx - Ax) * dx + (cy - Ay) * dy + (cz - Az) * dz) / L2
    if s >= 1.0:
        if i + 1 < len(SG) and SG[i + 1, 10] != 0:
            return pos  # the next segment holds the closest approach
        s = 1.0
    elif s <= 0.0:
        if SG[i, 10] != 0:
            px, py, pz = SG[i, 11], SG[i, 12], SG[i, 13]
            ex, ey, ez = Ax - px, Ay - py, Az - pz
            if (cx - px) * ex + (cy - py) * ey + (cz - pz) * ez < ex * ex + ey * ey + ez * ez:
                return pos
        s = 0.0
    Px = Ax + s * dx - cx
    Py = Ay + s * dy - cy
    Pz = Az + s * dz - cz
    dh = np.sqrt(Px * Px + Py * Py)
    if dh > reach:
        return pos
    kb = KB[i]
    if kb < 0:
        return pos
    tc = At + s * (SG[i, 8] - At)
    jc = kb + max(0, int(round(tc - ts[kb])))  # (the first sample may lie a fraction of a second after tc)
    if jc < 0 or jc >= len(run) or run[jc] != run[kb]:
        return pos
    # distant passes: the peak's time and rate change slowly, so one evaluation at the centre of each 2 x 2
    # block (anchored at even ix, iy) serves its four cells, carried to each cell centre by the gradients
    agg = False
    if dh > FAR_M:
        ax_ = ix - (ix & 1)
        ay_ = iy - (iy & 1)
        whole = ax_ + 1 < nx and ay_ + 1 < len(ground) and mask[ay_, ax_]
        if ax_ != ix or ay_ != iy:
            if whole and dh > FAR_M + 1.5 * g and dh < reach - 1.5 * g:
                return pos  # this block's anchor covers it
        elif whole:
            agg = True
            cx = cx + 0.5 * g
            cy = cy + 0.5 * g
            cz = 0.25 * (ground[iy, ix] + ground[iy, ix + 1] + ground[iy + 1, ix] + ground[iy + 1, ix + 1])
    vh = max(np.sqrt(V[jc, 0] ** 2 + V[jc, 1] ** 2), 30.0)  # (a take-off roll may start slow)
    R0 = np.sqrt(dh * dh + Pz * Pz)
    half = int(min(150.0, max(30.0, 0.025 * R0)))
    dur = R0 / vh
    step = 1 if dur < 10.0 else (2 if dur < 30.0 else 3)
    if dh < NEAR_CELLS * g:
        # close passes change too fast across the cell to be linearised: exact peaks at every sub-point
        for sp in range(n_sub * n_sub):
            qx = cx + ((sp % n_sub + 0.5) / n_sub - 0.5) * g
            qy = cy + ((sp // n_sub + 0.5) / n_sub - 0.5) * g
            npk = _scan(X, V, A, run, ts, jc, half, step, qx, qy, cz, c, wout, cand)
            for p in range(npk):
                th = wout[p, 0]
                lD = np.log(wout[p, 1])
                if th < th_lo or th > th_hi:
                    continue
                if len(lDk) > 0:
                    near = False
                    for u in range(len(lDk)):
                        if abs(lDk[u] - lD) <= lt:
                            near = True
                            break
                    if not near:
                        continue
                if pos < cap:
                    out[pos, 0] = iy * nx + ix
                    out[pos, 1] = th - th_lo
                    out[pos, 2] = 0.0
                    out[pos, 3] = 0.0
                    out[pos, 4] = 1.0
                    out[pos, 5] = lD
                    out[pos, 6] = 0.0
                    out[pos, 7] = 0.0
                    out[pos, 8] = 0.0
                    out[pos, 9] = wout[p, 2]
                    out[pos, 10] = sp
                pos += 1
        return pos
    npk = _scan(X, V, A, run, ts, jc, half, step, cx, cy, cz, c, wout, cand)
    e = 0.71 * g * (2.0 if agg else 1.0)
    h = g if agg else 0.5 * g
    for p in range(npk):
        th = wout[p, 0]
        Dq = wout[p, 1]
        js = int(wout[p, 2])
        if th < th_lo or th > th_hi:
            continue
        lD = np.log(Dq)
        _, Rq = _D(X, V, A, js, cx, cy, cz, c)
        lcell = max(np.log((Rq + e) / Rq), np.log(Rq / max(Rq - e, 30.0)))
        if len(lDk) > 0:
            near = False
            for u in range(len(lDk)):
                if abs(lDk[u] - lD) <= lt + lcell:
                    near = True
                    break
            if not near:
                continue
        # how the peak's time and height move across the cell (central differences over +/- half a cell); the
        # peak moves by about h / v along the track, so slow aircraft need a wider search
        win = int(min(40.0, 4.0 + 1.5 * h / vh))
        ok1, t1, D1 = _near_peak(X, V, A, run, ts, js, cx + h, cy, cz, c, cand, win)
        ok2, t2, D2 = _near_peak(X, V, A, run, ts, js, cx - h, cy, cz, c, cand, win)
        ok3, t3, D3 = _near_peak(X, V, A, run, ts, js, cx, cy + h, cz, c, cand, win)
        ok4, t4, D4 = _near_peak(X, V, A, run, ts, js, cx, cy - h, cz, c, cand, win)
        if ok1 and ok2 and ok3 and ok4 and D1 > 0 and D2 > 0 and D3 > 0 and D4 > 0:
            gx = (t1 - t2) / (2 * h)
            gy = (t3 - t4) / (2 * h)
            res = 1.0 + abs(0.5 * (t1 + t2) - th) + abs(0.5 * (t3 + t4) - th)  # second-order error at the edge
            gdx = (np.log(D1) - np.log(D2)) / (2 * h)
            gdy = (np.log(D3) - np.log(D4)) / (2 * h)
            lext = abs(0.5 * (np.log(D1) + np.log(D2)) - lD) + abs(0.5 * (np.log(D3) + np.log(D4)) - lD)
        else:
            # the peak is not found at all four: no gradient, so the whole cell's spread in time and rate
            gx = 0.0
            gy = 0.0
            res = e / vh + e / c + 1.0
            gdx = 0.0
            gdy = 0.0
            lext = lcell
        for q in range(4 if agg else 1):
            jx = ix + (q & 1) if agg else ix
            jy = iy + (q >> 1) if agg else iy
            if agg and not mask[jy, jx]:
                continue
            ox = (jx - ix - 0.5) * g if agg else 0.0  # cell centre relative to where it was evaluated
            oy = (jy - iy - 0.5) * g if agg else 0.0
            if pos < cap:
                out[pos, 0] = jy * nx + jx
                out[pos, 1] = th + gx * ox + gy * oy - th_lo
                out[pos, 2] = gx
                out[pos, 3] = gy
                out[pos, 4] = res
                out[pos, 5] = lD + gdx * ox + gdy * oy
                out[pos, 6] = gdx
                out[pos, 7] = gdy
                out[pos, 8] = lext
                out[pos, 9] = js
                out[pos, 10] = -1
            pos += 1
    return pos


FAR_M = 4000.0  # horizontal distances beyond this are evaluated once per 2 x 2 cells


@njit(cache=True, error_model="numpy")
def _interval(a, b, lo, hi):
    if abs(a) < 1e-12:
        if lo <= b <= hi:
            return -1e30, 1e30
        return 1.0, -1.0
    x1 = (lo - b) / a
    x2 = (hi - b) / a
    if x1 <= x2:
        return x1, x2
    return x2, x1


@njit(cache=True, error_model="numpy")
def _segment_peaks(i, x0, y0, nx, ny, g, ground, mask, SG, KB, X, V, A, run, ts, lDk, lt, reach, c, th_lo, th_hi,
                   out, pos, cap, wout, cand, n_sub):
    """Visit the cells whose closest point on this track is on segment i (within reach), make their items."""
    Ax, Ay = SG[i, 0], SG[i, 1]
    Bx, By = SG[i, 5], SG[i, 6]
    dxh = Bx - Ax
    dyh = By - Ay
    Lh2 = dxh * dxh + dyh * dyh
    if Lh2 < 1.0:
        return pos
    Lh = np.sqrt(Lh2)
    has_p = SG[i, 10] != 0
    has_n = i + 1 < len(SG) and SG[i + 1, 10] != 0
    pad = 0.75 * g
    H = reach + pad
    s_lo = -pad / Lh if has_p else -H / Lh
    s_hi = 1.0 + pad / Lh if has_n else 1.0 + H / Lh
    pxv, pyv = SG[i, 11], SG[i, 12]
    exh = Ax - pxv
    eyh = Ay - pyv
    Lp2 = exh * exh + eyh * eyh
    a_s = dxh / Lh2
    a_p = -dyh / Lh
    ext = H + (H if (not has_p or not has_n) else 0.0)
    iy0 = max(0, int(np.floor((min(Ay, By) - ext - y0) / g)))
    iy1 = min(ny - 1, int(np.floor((max(Ay, By) + ext - y0) / g)))
    for iy in range(iy0, iy1 + 1):
        cy = y0 + (iy + 0.5) * g
        b_s = (-Ax * dxh + (cy - Ay) * dyh) / Lh2
        b_p = (Ax * dyh + (cy - Ay) * dxh) / Lh
        p_lo, p_hi = _interval(a_p, b_p, -H, H)
        for st in range(2):
            if st == 0:
                w_lo, w_hi = _interval(a_s, b_s, s_lo, s_hi)
            else:
                # outside of a turn at vertex A: before this segment but past the previous one
                if not has_p or Lp2 < 1.0:
                    continue
                w_lo, w_hi = _interval(a_s, b_s, -H / Lh, -pad / Lh)
                a_q = exh / Lp2
                b_q = (-pxv * exh + (cy - pyv) * eyh) / Lp2
                q_lo, q_hi = _interval(a_q, b_q, 1.0 - pad / np.sqrt(Lp2), 1e30)
                w_lo = max(w_lo, q_lo)
                w_hi = min(w_hi, q_hi)
            xa = max(w_lo, p_lo)
            xb = min(w_hi, p_hi)
            if xa > xb:
                continue
            ix0 = max(0, int(np.ceil((xa - x0) / g - 0.5)))
            ix1 = min(nx - 1, int(np.floor((xb - x0) / g - 0.5)))
            for ix in range(ix0, ix1 + 1):
                if st == 1:
                    cx = x0 + (ix + 0.5) * g
                    if a_s * cx + b_s >= -pad / Lh:
                        continue  # already visited by the main strip
                if mask[iy, ix]:
                    pos = _cell_peaks(i, ix, iy, x0, y0, nx, ny, g, ground, mask, SG, KB, X, V, A, run, ts, lDk, lt,
                                      reach, c, th_lo, th_hi, out, pos, cap, wout, cand, n_sub)
    return pos


@njit(parallel=True, cache=True, error_model="numpy")
def block_peaks(segs_idx, x0, y0, nx, ny, g, ground, mask, SG, KB, X, V, A, run, ts, lDk, lt, reach, c, th_lo, th_hi,
                n_sub, nch, cap):
    """All peak rows of a block. Chunk ch handles segments ch, ch + nch, ... and writes rows ch*cap ..; the
    counts per chunk are returned (above cap: that chunk overflowed and the block must be redone)."""
    out = np.empty((nch * cap, NF), np.float32)
    counts = np.zeros(nch, np.int64)
    n = len(segs_idx)
    for ch in prange(nch):
        wout = np.zeros((NPK, 3))
        cand = np.zeros((128, 3))
        sub = out[ch * cap:(ch + 1) * cap]
        pos = 0
        for jj in range(ch, n, nch):
            pos = _segment_peaks(segs_idx[jj], x0, y0, nx, ny, g, ground, mask, SG, KB, X, V, A, run, ts, lDk, lt,
                                 reach, c, th_lo, th_hi, sub, pos, cap, wout, cand, n_sub)
        counts[ch] = pos
    return out, counts


# --------------------------------------------------------------------------- the join

@njit(cache=True, error_model="numpy")
def _cell_join(P, a, b, PKt, PKtol, lDk, lt, anchor_rank, n_anchor, m_min, g, n_sub, K, PKk, out, pos, cap,
               mu, mt, mr, ms, key, order):
    """Coincidences in one cell (rows P[a:b], sorted by cell). Every pass match is (pass, start time at the cell
    centre, row). Only matches of anchor passes start a search (a candidate with >= m_min votes must contain one
    of the Kp - m_min + 1 anchors); at each of the n_sub^2 points the other passes vote if their start time
    window overlaps the anchor's and their predicted rate fits. Each coincidence is emitted once, by the lowest
    ranked anchor in it. Output row: cell, point, T0 (rel.), votes, score, then K row indices (-1 = none)."""
    Kp = len(PKt)
    h = 0.5 * g
    lt0 = lt
    M = 0
    for r in range(a, b):
        lD = P[r, 5]
        lspr = (abs(P[r, 6]) + abs(P[r, 7])) * h
        for u in range(Kp):
            if abs(lDk[u] - lD) - lspr <= lt + P[r, 8]:
                mu[M] = u
                mt[M] = P[r, 1] - PKt[u]
                mr[M] = r
                ms[M] = (abs(P[r, 2]) + abs(P[r, 3])) * h + P[r, 4] + PKtol[u]
                M += 1
    if M == 0:
        return pos
    for q in range(M):
        key[q] = mu[q] * 1.0e7 + mt[q]
    o = np.argsort(key[:M])
    for q in range(M):
        order[q] = o[q]
    # per-pass ranges in the sorted order and the widest spread of each pass
    starts = np.zeros(Kp + 1, np.int64)
    smax = np.zeros(Kp)
    for q in range(M):
        u = mu[order[q]]
        starts[u + 1] += 1
        if ms[order[q]] > smax[u]:
            smax[u] = ms[order[q]]
    for u in range(Kp):
        starts[u + 1] += starts[u]
    have = np.zeros(Kp, np.bool_)
    bestr = np.full(Kp, -1, np.int64)
    bestq = np.zeros(Kp)
    bT = np.zeros(Kp)
    keep_r = np.full(Kp, -1, np.int64)
    cu = np.empty(256, np.int64)  # (pass, row) pairs that can agree with the anchor somewhere in the cell
    cr = np.empty(256, np.int64)
    for u_a in range(Kp):
        if anchor_rank[u_a] >= n_anchor:
            continue
        for qa in range(starts[u_a], starts[u_a + 1]):
            ia = order[qa]
            ta = mt[ia]
            sa = ms[ia]
            ra = mr[ia]
            spa = int(P[ra, 10])
            wa = P[ra, 4] + PKtol[u_a]
            # cell level, exact for linear times: some point of the cell where the two windows overlap
            nc = 0
            cnt = 1
            for u in range(Kp):
                if u == u_a:
                    continue
                lo_t = ta - sa - smax[u]
                lo = starts[u]
                hi = starts[u + 1]
                while lo < hi:
                    md = (lo + hi) // 2
                    if mt[order[md]] < lo_t:
                        lo = md + 1
                    else:
                        hi = md
                found = False
                q = lo
                while q < starts[u + 1] and mt[order[q]] <= ta + sa + smax[u]:
                    r = mr[order[q]]
                    q += 1
                    if spa >= 0 and P[r, 10] >= 0 and int(P[r, 10]) != spa:
                        continue
                    dT = mt[order[q - 1]] - ta
                    spread = (abs(P[r, 2] - P[ra, 2]) + abs(P[r, 3] - P[ra, 3])) * h
                    if abs(dT) - spread > P[r, 4] + PKtol[u] + wa:
                        continue
                    if nc < 256:
                        cu[nc] = u
                        cr[nc] = r
                        nc += 1
                    found = True
                if found:
                    cnt += 1
            if cnt < m_min:
                continue
            best_v = 0
            best_s = 0.0
            best_p = -1
            best_T = 0.0
            for sp in range(n_sub * n_sub):
                if spa >= 0 and sp != spa:
                    continue
                sy = sp // n_sub
                sx = sp % n_sub
                ox = ((sx + 0.5) / n_sub - 0.5) * g
                oy = ((sy + 0.5) / n_sub - 0.5) * g
                miss = abs(lDk[u_a] - (P[ra, 5] + P[ra, 6] * ox + P[ra, 7] * oy))
                tl = lt0 + P[ra, 8]
                if miss > tl:
                    continue
                Ta = P[ra, 1] + P[ra, 2] * ox + P[ra, 3] * oy - PKt[u_a]
                for u in range(Kp):
                    have[u] = False
                    bestr[u] = -1
                    bestq[u] = 0.0
                have[u_a] = True
                bestr[u_a] = ra
                bestq[u_a] = 1.0 - 0.3 * miss / tl
                bT[u_a] = Ta
                for e_ in range(nc):
                    u = cu[e_]
                    r = cr[e_]
                    if P[r, 10] >= 0 and int(P[r, 10]) != sp:
                        continue
                    m2 = abs(lDk[u] - (P[r, 5] + P[r, 6] * ox + P[r, 7] * oy))
                    t2 = lt0 + P[r, 8]
                    if m2 > t2:
                        continue
                    Tu = P[r, 1] + P[r, 2] * ox + P[r, 3] * oy - PKt[u]
                    wu = P[r, 4] + PKtol[u] + wa
                    dT = abs(Tu - Ta)
                    if dT > wu:
                        continue
                    # score: rate and timing agreement
                    qq = 1.0 - 0.3 * m2 / t2 - 0.3 * dT / wu
                    if qq > bestq[u]:
                        bestq[u] = qq
                        bestr[u] = r
                        bT[u] = Tu
                        have[u] = True
                # one common start time: from the anchor's window, add passes by score while the windows still
                # share a point (pairwise agreement with the anchor alone is not enough)
                L = Ta - (P[ra, 4] + PKtol[u_a])
                Rr = Ta + (P[ra, 4] + PKtol[u_a])
                v = 1
                sc = bestq[u_a]
                for _ in range(Kp):
                    bu = -1
                    for u in range(Kp):
                        if have[u] and u != u_a and bestq[u] > 0 and (bu < 0 or bestq[u] > bestq[bu]):
                            bu = u
                    if bu < 0:
                        break
                    w0 = P[bestr[bu], 4] + PKtol[bu]
                    lo2 = max(L, bT[bu] - w0)
                    hi2 = min(Rr, bT[bu] + w0)
                    if lo2 <= hi2:
                        L, Rr = lo2, hi2
                        v += 1
                        sc += bestq[bu]
                        bestq[bu] = -bestq[bu]  # taken
                    else:
                        have[bu] = False
                for u in range(Kp):
                    if bestq[u] < 0:
                        bestq[u] = -bestq[u]
                Tsum = 0.5 * (L + Rr) * v
                if v > best_v or (v == best_v and sc > best_s):
                    # emitted by the lowest ranked anchor that is part of it
                    dup = False
                    for u in range(Kp):
                        if have[u] and anchor_rank[u] < anchor_rank[u_a]:
                            dup = True
                            break
                    if dup:
                        continue
                    best_v = v
                    best_s = sc
                    best_p = sp
                    best_T = Tsum / v
                    for u in range(Kp):
                        keep_r[u] = bestr[u]
            if best_v >= m_min:
                if pos < cap:
                    out[pos, 0] = P[a, 0]
                    out[pos, 1] = best_p
                    out[pos, 2] = best_T
                    out[pos, 3] = best_v
                    out[pos, 4] = best_s
                    for k in range(K):
                        out[pos, 5 + k] = -1
                    for u in range(Kp):
                        out[pos, 5 + PKk[u]] = keep_r[u]
                pos += 1
    return pos


@njit(parallel=True, cache=True, error_model="numpy")
def join_block(P, cstart, PKt, PKtol, lDk, lt, anchor_rank, n_anchor, m_min, g, n_sub, K, PKk, nch, cap):
    """Coincidences of a block's rows P (sorted by cell; cstart = row offsets of the cells present)."""
    out = np.empty((nch * cap, 5 + K))
    counts = np.zeros(nch, np.int64)
    nc = len(cstart) - 1
    Kp = len(PKt)
    mx = 1
    for ci in range(nc):
        mx = max(mx, (cstart[ci + 1] - cstart[ci]) * Kp)
    for ch in prange(nch):
        mu = np.empty(mx, np.int64)
        mt = np.empty(mx)
        mr = np.empty(mx, np.int64)
        ms = np.empty(mx)
        key = np.empty(mx)
        order = np.empty(mx, np.int64)
        sub = out[ch * cap:(ch + 1) * cap]
        pos = 0
        for ci in range(ch, nc, nch):
            pos = _cell_join(P, cstart[ci], cstart[ci + 1], PKt, PKtol, lDk, lt, anchor_rank, n_anchor, m_min, g,
                             n_sub, K, PKk, sub, pos, cap, mu, mt, mr, ms, key, order)
        counts[ch] = pos
    return out, counts


# --------------------------------------------------------------------------- driver

def _chunked(fn, args, nch, cap):
    """Run a chunked kernel, growing the per-chunk capacity until nothing overflows; rows of all chunks."""
    while True:
        out, counts = fn(*args, nch, cap)
        if counts.max() <= cap:
            return np.concatenate([out[ch * cap: ch * cap + counts[ch]] for ch in range(nch)]), cap
        cap = int(counts.max() * 1.3) + 1024


def run(events, day_dirs, cfg: KinConfig):
    """Returns (candidates, flights, points, projection), candidates None when nothing explains enough passes."""
    log = cfg.log
    T = time.time()
    proj = LocalProjection(cfg.bbox)
    K = len(events)
    t_lo = cfg.t0_range[0] - 900
    t_hi = cfg.t0_range[1] + cfg.audio_duration + 900
    pad_deg = (cfg.reach_m + 5000.0) / 111000.0
    fl, pts = adsb.load(day_dirs, bbox=_pad_bbox(cfg.bbox, pad_deg), t_range=(t_lo, t_hi), classes=cfg.classes)
    if pts.empty:
        raise RuntimeError("no ADS-B points in the search window/area")
    segs = Segments(pts, proj, cfg.max_seg_gap_s, cfg.simplify_m)
    SG = _seg_table(segs)
    KN = Kinematics(pts, proj)
    KB = KN.index(segs.fid, segs.at)
    t_load = time.time() - T

    AP = [peaks.audio_peak(e) for e in events]
    pk = np.array([k for k in range(K) if AP[k].ok], np.int64)
    PKt = np.array([AP[k].t for k in pk], np.float64)
    PKD = np.array([AP[k].rate for k in pk], np.float64)
    PKtol = np.array([AP[k].tol_t for k in pk], np.float64)
    Kp = len(pk)
    m_min = max(min(cfg.min_votes, Kp), int(np.ceil(cfg.vote_frac * Kp)))
    # anchors: the passes timed most precisely; any candidate with >= m_min votes contains one of them
    n_anchor = Kp - m_min + 1
    anchor_rank = np.empty(Kp, np.int64)
    anchor_rank[np.argsort(PKtol, kind="stable")] = np.arange(Kp)
    log(f"loaded {len(fl)} flights, {segs.n} segments, {len(KN.t)} flight seconds ({t_load:.1f}s); "
        f"{Kp} of {K} passes have a clear fastest fall; candidates need {m_min} of them ({n_anchor} anchors)")
    if Kp < 2:
        return None, fl, pts, proj

    g = cfg.cell_m
    ns = cfg.n_sub
    dem = DEM(cfg.dem_zoom)
    gcfg = _GroundCfg(cfg, g)
    s_idx, b_key = segs.block_index(np.ones(segs.n, bool), cfg.block_m, cfg.reach_m + g)
    groups = _groups(b_key)
    t0_lo, t0_hi = map(float, cfg.t0_range)
    th_lo = t0_lo + PKt.min() - 60.0
    th_hi = t0_hi + PKt.max() + 60.0
    lDk = np.log(PKD)
    lt = float(np.log(cfg.rate_tol))
    cache = {}
    res = dict(x=[], y=[], T0=[], votes=[], score=[], fid=[])
    n_rows = n_cand = 0
    tp = tj = 0.0
    nch = 256
    capP = 8192
    capJ = 1024
    probe_bk = None
    if cfg.probe is not None:
        from .search import _bkey
        probe_bk = int(_bkey(int(np.floor(cfg.probe[0] / cfg.block_m)), int(np.floor(cfg.probe[1] / cfg.block_m))))
    for bk in sorted(groups):
        x0, y0, nx, ny, mask, ground = _ground(bk, gcfg, proj, dem, cache)
        if not mask.any():
            continue
        sidx = s_idx[groups[bk]].astype(np.int64)
        tq = time.time()
        P, capP = _chunked(block_peaks, (sidx, x0, y0, nx, ny, g, ground, mask, SG, KB, KN.X, KN.V, KN.A, KN.run,
                                         KN.t, lDk, lt, cfg.reach_m, cfg.c, th_lo, th_hi, ns), nch, capP)
        P = P.astype(np.float64)
        P = P[np.argsort(P[:, 0], kind="stable")]
        tp += time.time() - tq
        n_rows += len(P)
        if len(P) == 0:
            continue
        tq = time.time()
        cells = P[:, 0].astype(np.int64)
        cstart = np.r_[np.flatnonzero(np.r_[True, cells[1:] != cells[:-1]]), len(P)].astype(np.int64)
        J, capJ = _chunked(join_block, (P, cstart, PKt - (th_lo - t0_lo), PKtol, lDk, lt, anchor_rank, n_anchor,
                                        m_min, g, ns, K, pk), nch, capJ)
        tj += time.time() - tq
        if cfg.probe is not None and bk == probe_bk:
            _probe_block(cfg, P, J, x0, y0, nx, g, t0_lo, (PKt, PKtol, lDk, lt, th_lo), log)
        if len(J) == 0:
            continue
        n_cand += len(J)
        c_ = J[:, 0].astype(np.int64)
        sp = J[:, 1].astype(np.int64)
        iy, ix = np.divmod(c_, nx)
        sy, sx = np.divmod(sp, ns)
        res["x"].append(x0 + ix * g + (sx + 0.5) * g / ns)
        res["y"].append(y0 + iy * g + (sy + 0.5) * g / ns)
        res["T0"].append(J[:, 2] + t0_lo)
        res["votes"].append(J[:, 3].astype(np.int64))
        res["score"].append(J[:, 4])
        rr = J[:, 5:].astype(np.int64)
        res["fid"].append(np.where(rr >= 0, KN.fid[P[np.maximum(rr, 0), 9].astype(np.int64)], -1).astype(np.int32))
    log(f"  peak rows: {n_rows} ({tp:.1f}s); coincidences: {n_cand} ({tj:.1f}s)")
    if not res["T0"]:
        return None, fl, pts, proj
    res = {k: np.concatenate(v) for k, v in res.items()}
    order = np.lexsort((-res["score"], -res["votes"]))
    if cfg.probe is not None:
        near = (np.hypot(res["x"] - cfg.probe[0], res["y"] - cfg.probe[1]) < 300) & (np.abs(res["T0"] - cfg.probe[2]) < 30)
        rank = np.empty(len(order), np.int64)
        rank[order] = np.arange(len(order))
        log(f"  probe: {int(near.sum())} candidates near it; best ranks {np.sort(rank[near])[:5].tolist()} of "
            f"{len(rank)}; their votes {res['votes'][near][:5].tolist()}")
    order = order[: cfg.keep]
    res = {k: v[order] for k, v in res.items()}
    vcnt = np.bincount(res["votes"], minlength=Kp + 1)
    log(f"  {len(res['T0'])} candidates kept, by votes {dict((v, int(vcnt[v])) for v in range(len(vcnt)) if vcnt[v])} "
        f"(total {time.time() - T:.1f}s)")
    bank = TrajectoryBank(pts, fl.fid.to_numpy())
    cands = dict(x=res["x"], y=res["y"], T0=res["T0"], votes=res["votes"], coarse=res["score"], fid=res["fid"],
                 bank=bank)
    return cands, fl, pts, proj


def _probe_block(cfg, P, J, x0, y0, nx, g, t0_lo, passes, log):
    """Debugging: how the known answer's cell fares (rows near it per pass, coincidences found there)."""
    pk_t, pk_tol, lDk, lt, th_lo = passes
    px, py, pT = cfg.probe
    c_ = int((py - y0) // g) * nx + int((px - x0) // g)
    sel = J[:, 0].astype(np.int64) == c_
    near = sel & (np.abs(J[:, 2] + t0_lo - pT) < 30)
    log(f"  probe cell: {int((P[:, 0] == c_).sum())} peak rows; coincidences in the cell {int(sel.sum())}, "
        f"near the true start {int(near.sum())}, their votes {J[near, 3].astype(int).tolist()[:6]}")
    ox = px - (x0 + ((px - x0) // g + 0.5) * g)
    oy = py - (y0 + ((py - y0) // g + 0.5) * g)
    R = P[P[:, 0] == c_]
    T = R[:, 1] + R[:, 2] * ox + R[:, 3] * oy  # heard time at the truth (rel. to th_lo)
    lD = R[:, 5] + R[:, 6] * ox + R[:, 7] * oy
    for u in range(len(pk_t)):
        dt = T - (pk_t[u] + pT - th_lo)
        dl = lDk[u] - lD
        score = np.abs(dt) / (R[:, 4] + pk_tol[u]) + np.abs(dl) / (lt + R[:, 8])
        j = int(np.argmin(score))
        log(f"    pass {u}: closest row dt {dt[j]:+.1f}s (allowed {R[j, 4] + pk_tol[u]:.1f}), "
            f"dlnD {dl[j]:+.2f} (allowed {lt + R[j, 8]:.2f}), gradient {R[j, 2]*1000:+.1f},{R[j, 3]*1000:+.1f} s/km")
