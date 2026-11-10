"""Trajectories packed for compiled code, and the exact Doppler curve a place would hear from them.

* TrajectoryBank: every needed flight in flat arrays (knot times, ECEF position, ECEF velocity).
* _curve: heard frequency factor at regular observer times, solving for the emission time.
* line_items: how well that curve fits a pass's tracked line, for a range of start-time shifts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit, prange

from . import adsb
from .dem import DEM
from .geo import ecef, enu_to_ecef_vec, geoid_undulation
from .measure import Event


# --------------------------------------------------------------------------- packed trajectories

class TrajectoryBank:
    """All needed flights packed into flat arrays (knot times, ECEF position, ECEF velocity),
    indexed by `start` (CSR) so compiled code can evaluate any of them."""

    def __init__(self, pts: pd.DataFrame, fids):
        fids = np.unique(np.asarray(fids))
        p = pts[pts.fid.isin(fids)].sort_values(["fid", "t"], kind="stable")
        h = adsb.fill_geometric_altitude(p)
        t = p.t.to_numpy()
        fid = p.fid.to_numpy()
        keep = np.r_[True, (np.diff(t) > 0.05) | (fid[1:] != fid[:-1])]
        p, t, fid, h = p[keep], t[keep], fid[keep], h[keep]
        lat, lon = p.lat.to_numpy(), p.lon.to_numpy()
        pos = ecef(lat, lon, h)
        gs = p.gs.to_numpy(np.float64)
        trk = np.radians(p.track.to_numpy(np.float64))
        vr = np.nan_to_num(p.vrate.to_numpy(np.float64))
        vel = enu_to_ecef_vec(lat, lon, gs * np.sin(trk), gs * np.cos(trk), vr)
        # finite-difference fallback for missing speed/track
        fd = np.zeros_like(pos)
        same_next = np.r_[fid[1:] == fid[:-1], False]
        same_prev = np.r_[False, fid[1:] == fid[:-1]]
        nxt = np.minimum(np.arange(len(t)) + 1, len(t) - 1)
        prv = np.maximum(np.arange(len(t)) - 1, 0)
        a = np.where(same_prev, prv, np.arange(len(t)))
        b = np.where(same_next, nxt, np.arange(len(t)))
        dt = np.maximum(t[b] - t[a], 1e-3)[:, None]
        fd = (pos[b] - pos[a]) / dt
        bad = ~np.isfinite(vel).all(axis=1)
        vel[bad] = fd[bad]
        u, start = np.unique(fid, return_index=True)
        self.fids = u
        self.start = np.r_[start, len(fid)].astype(np.int64)
        self.t = t.astype(np.float64)
        self.pos = np.ascontiguousarray(pos)
        self.vel = np.ascontiguousarray(vel)

    def index(self, fids):
        i = np.searchsorted(self.fids, fids)
        return i.astype(np.int64)


@njit(cache=True, error_model="numpy")
def _herm(tk, P, V, a, b, tau, out_p, out_v):
    """Position & velocity of the flight with knots a..b-1 at tau (linear, clamped)."""
    if tau <= tk[a]:
        i = a
        s = 0.0
    elif tau >= tk[b - 1]:
        i = b - 2
        s = 1.0
    else:
        lo, hi = a, b - 1
        while hi - lo > 1:
            m = (lo + hi) // 2
            if tk[m] <= tau:
                lo = m
            else:
                hi = m
        i = lo
        s = (tau - tk[i]) / (tk[i + 1] - tk[i])
    for k in range(3):
        out_p[k] = P[i, k] + s * (P[i + 1, k] - P[i, k])
        out_v[k] = V[i, k] + s * (V[i + 1, k] - V[i, k])


@njit(cache=True, error_model="numpy")
def _curve(tk, P, V, a, b, o, t_first_obs, step, n, c, g, ok):
    """Doppler factor g at observer times t_first_obs + j*step, j < n (emission-time exact)."""
    pp = np.empty(3)
    vv = np.empty(3)
    tau = t_first_obs
    for j in range(n):
        t_obs = t_first_obs + j * step
        tau = min(tau + step, t_obs)  # warm start from previous emission time
        for _ in range(4):
            _herm(tk, P, V, a, b, tau, pp, vv)
            r = np.sqrt((pp[0] - o[0]) ** 2 + (pp[1] - o[1]) ** 2 + (pp[2] - o[2]) ** 2)
            tau = t_obs - r / c
        if tau < tk[a] or tau > tk[b - 1]:
            ok[j] = False
            g[j] = 1.0
            continue
        _herm(tk, P, V, a, b, tau, pp, vv)
        dx, dy, dz = pp[0] - o[0], pp[1] - o[1], pp[2] - o[2]
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        g[j] = c / (c + (vv[0] * dx + vv[1] * dy + vv[2] * dz) / r)
        ok[j] = True


@njit(parallel=True, cache=True, error_model="numpy")
def line_items(obs, item_obs, item_traj, item_T0, tk, P, V, start, la, lf, lmask, shifts, f0_lo, f0_hi, sigma, c):
    """Fast profile: how well the model curve f0*g(t) fits the tracked line.

    la: regular audio-time grid (same step as shifts); lf/lmask: line frequency where seen.
    For each shift the best f0 has a closed form (least squares, then one robust re-weighting),
    clipped to [f0_lo, f0_hi]. Score = mean over seen line points of exp(-0.5 (res/sigma)^2),
    in [0, 1]; points whose emission time is outside the flight's data count as misses.
    """
    n = len(item_obs)
    ns = len(shifts)
    na = len(la)
    step = la[1] - la[0]
    nu = na + ns - 1
    nseen = 0
    for j in range(na):
        if lmask[j]:
            nseen += 1
    out = np.zeros((n, ns))
    for it in prange(n):
        a = start[item_traj[it]]
        b = start[item_traj[it] + 1]
        if b - a < 2:
            continue
        g = np.empty(nu)
        ok = np.zeros(nu, np.bool_)
        _curve(tk, P, V, a, b, obs[item_obs[it]], item_T0[it] + shifts[0] + la[0], step, nu, c, g, ok)
        for si in range(ns):
            num = 0.0
            den = 0.0
            for j in range(na):
                u = j + si
                if lmask[j] and ok[u]:
                    num += lf[j] * g[u]
                    den += g[u] * g[u]
            if den == 0:
                continue
            f0 = min(max(num / den, f0_lo), f0_hi)
            num = 0.0
            den = 0.0
            for j in range(na):
                u = j + si
                if lmask[j] and ok[u]:
                    res = lf[j] - f0 * g[u]
                    w = 1.0 / (1.0 + (res / (3 * sigma)) ** 2)
                    num += w * lf[j] * g[u]
                    den += w * g[u] * g[u]
            if den > 0:
                f0 = min(max(num / den, f0_lo), f0_hi)
            tot = 0.0
            for j in range(na):
                u = j + si
                if lmask[j] and ok[u]:
                    res = (lf[j] - f0 * g[u]) / sigma
                    tot += np.exp(-0.5 * res * res)
            out[it, si] = tot / nseen
    return out


def line_grid(e: Event, step):
    """Event line resampled onto a regular grid of `step` seconds (mask where not seen)."""
    t = np.asarray(e.line_t, float)
    f = np.asarray(e.line_f, float)
    la = np.arange(np.ceil(t[0] / step) * step, t[-1] + 1e-9, step)
    k = np.clip(np.searchsorted(t, la), 1, len(t) - 1)
    near = np.minimum(np.abs(t[k] - la), np.abs(t[k - 1] - la))
    lf = np.interp(la, t, f)
    return la, lf, near <= step


# --------------------------------------------------------------------------- driver

def observers(lat, lon, dem: DEM, obs_height=1.5):
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    elev = dem.elevation(lat, lon)
    # geoid varies slowly: evaluate on rounded coordinates
    key = np.round(lat, 1) * 10000 + np.round(lon, 1)
    uk, inv = np.unique(key, return_inverse=True)
    first = np.unique(inv, return_index=True)[1]
    N = geoid_undulation(lat[first], lon[first])[inv]
    return ecef(lat, lon, elev + N + obs_height)


@njit(cache=True, error_model="numpy")
def curve_walk(tk, P, V, a, b, o, t_first, step, n, c, g, ok):
    """Doppler factor at observer times t_first + j*step (emission time solved; positions linear between knots)."""
    # start at the knot just before the first observer time (binary search), then walk
    lo = a
    hi = b - 1
    while hi - lo > 1:
        m = (lo + hi) // 2
        if tk[m] <= t_first:
            lo = m
        else:
            hi = m
    i = lo
    tau = t_first
    gp = 1.0
    for j in range(n):
        t_obs = t_first + j * step
        # emission time advances by step * g (g = dtau / dt_obs); then one correction is enough
        tau = min(tau + step * gp, t_obs) if j > 0 else t_obs
        s = 0.0
        for it in range(3 if j == 0 else 1):
            if tau <= tk[a]:
                i = a
                s = 0.0
            elif tau >= tk[b - 1]:
                i = b - 2
                s = 1.0
            else:
                while i < b - 2 and tk[i + 1] <= tau:
                    i += 1
                while i > a and tk[i] > tau:
                    i -= 1
                s = (tau - tk[i]) / (tk[i + 1] - tk[i])
            px = P[i, 0] + s * (P[i + 1, 0] - P[i, 0]) - o[0]
            py = P[i, 1] + s * (P[i + 1, 1] - P[i, 1]) - o[1]
            pz = P[i, 2] + s * (P[i + 1, 2] - P[i, 2]) - o[2]
            tau = t_obs - np.sqrt(px * px + py * py + pz * pz) / c
        if tau < tk[a] or tau > tk[b - 1]:
            ok[j] = False
            g[j] = 1.0
            continue
        while i < b - 2 and tk[i + 1] <= tau:
            i += 1
        while i > a and tk[i] > tau:
            i -= 1
        s = (tau - tk[i]) / (tk[i + 1] - tk[i])
        px = P[i, 0] + s * (P[i + 1, 0] - P[i, 0]) - o[0]
        py = P[i, 1] + s * (P[i + 1, 1] - P[i, 1]) - o[1]
        pz = P[i, 2] + s * (P[i + 1, 2] - P[i, 2]) - o[2]
        vx = V[i, 0] + s * (V[i + 1, 0] - V[i, 0])
        vy = V[i, 1] + s * (V[i + 1, 1] - V[i, 1])
        vz = V[i, 2] + s * (V[i + 1, 2] - V[i, 2])
        rr = np.sqrt(px * px + py * py + pz * pz)
        g[j] = c / (c + (vx * px + vy * py + vz * pz) / rr)
        gp = g[j]
        ok[j] = True
