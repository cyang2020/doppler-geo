"""Fine scoring with the real trajectories.

A candidate is a place and a recording start time T0. For every pass, every flight near the place
is tried: the exact Doppler curve that flight would produce there is fitted to the pass's tracked
line (rest frequency free) for small shifts of T0. Per pass the best flight counts; the candidate's
score is the best common shift of the sum over passes, so all passes must line up with one start time.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from . import score as sc


def score_obs(lat, lon, dem):
    return sc.observers(np.asarray(lat, float), np.asarray(lon, float), dem)


def best_flights(lat, lon, T0, events, bank, dem, c, shift_s=6.0, step_s=0.5, sigma_hz=1.5):
    """For one place and start time: for each pass, the flight (of all in the bank) that fits best.
    Returns (fid per pass or -1, score per pass, best T0 shift per pass)."""
    obs = score_obs([lat], [lon], dem)
    nf = len(bank.fids)
    shifts = np.arange(-shift_s, shift_s + 1e-9, step_s)
    out_f, out_s, out_t = [], [], []
    for e in events:
        la, lf, lm = sc.line_grid(e, step_s)
        pad = 0.03 * e.f0
        prof = sc.line_items(obs, np.zeros(nf, np.int64), np.arange(nf, dtype=np.int64), np.full(nf, float(T0)),
                                bank.t, bank.pos, bank.vel, bank.start, la, lf, lm, shifts,
                                min(lf[lm]) * 0.85 - pad, max(lf[lm]) * 1.15 + pad, sigma_hz, c)
        m = prof.max(axis=1)
        i = int(m.argmax())
        out_f.append(int(bank.fids[i]) if m[i] > 0 else -1)
        out_s.append(float(m[i]))
        out_t.append(float(shifts[int(prof[i].argmax())]))
    return np.array(out_f), np.array(out_s), np.array(out_t)


# --------------------------------------------------------------------------- full scoring

from .score import curve_walk as _curve_walk  # noqa: E402


@njit(cache=True, error_model="numpy")
def _near_flights(cx, cy, t_lo, t_hi, R2, px, py, pt, pf, n_fl, stamp, mark, out):
    """Flights (bank indices) with a point within sqrt(R2) of (cx, cy) between t_lo and t_hi."""
    a = np.searchsorted(pt, t_lo)
    b = np.searchsorted(pt, t_hi)
    n = 0
    for i in range(a, b):
        dx = px[i] - cx
        dy = py[i] - cy
        if dx * dx + dy * dy <= R2:
            f = pf[i]
            if mark[f] != stamp:
                mark[f] = stamp
                out[n] = f
                n += 1
    return n


@njit(parallel=True, cache=True, error_model="numpy")
def full_scores(obs, cxy, T0, LA, LF, LI, eoff, ioff, nseen, f0lo, f0hi, win, R, tk, P, V, start, px, py, pt, pf,
                shift_s, step, sigma, c):
    """Every candidate, every pass, every flight near the candidate: real Doppler curve vs the pass's line.
    Per pass the best flight is taken at each start-time shift; the candidate's score is the best shift of
    the sum over passes. Returns total, best T0, per-pass score and per-pass flight (bank index or -1)."""
    n = len(T0)
    K = len(eoff) - 1
    ns = int(2 * shift_s / step) + 1
    total = np.zeros(n)
    bestT0 = np.zeros(n)
    parts = np.zeros((n, K))
    flights = np.full((n, K), -1, np.int64)
    n_fl = len(start) - 1
    for ci in prange(n):
        mark = np.zeros(n_fl, np.int64)
        near = np.empty(n_fl, np.int64)
        Pk = np.zeros((K, ns))
        Fk = np.full((K, ns), -1, np.int64)
        t_first = T0[ci] - shift_s
        for k in range(K):
            e0 = eoff[k]
            na = eoff[k + 1] - e0
            nf = _near_flights(cxy[ci, 0], cxy[ci, 1], T0[ci] + win[k, 0] - 120.0, T0[ci] + win[k, 1] + 60.0,
                               R[k] * R[k], px, py, pt, pf, n_fl, k + 1, mark, near)
            nu = na + ns - 1
            g = np.empty(nu)
            okg = np.zeros(nu, np.bool_)
            for q in range(nf):
                f = near[q]
                a = start[f]
                b = start[f + 1]
                if b - a < 2:
                    continue
                _curve_walk(tk, P, V, a, b, obs[ci], t_first + LA[e0], step, nu, c, g, okg)
                for j in range(ns):
                    num = 0.0
                    den = 0.0
                    for u_ in range(ioff[k], ioff[k + 1]):
                        m = LI[u_]
                        u = m + j
                        if okg[u]:
                            num += LF[e0 + m] * g[u]
                            den += g[u] * g[u]
                    if den == 0:
                        continue
                    f0 = min(max(num / den, f0lo[k]), f0hi[k])
                    tot = 0.0
                    for u_ in range(ioff[k], ioff[k + 1]):
                        m = LI[u_]
                        u = m + j
                        if okg[u]:
                            x = (LF[e0 + m] - f0 * g[u]) / (2.0 * sigma)
                            x2 = x * x
                            if x2 < 1.0:
                                tot += (1.0 - x2) * (1.0 - x2)
                    tot /= nseen[k]
                    if tot > Pk[k, j]:
                        Pk[k, j] = tot
                        Fk[k, j] = f
        bj = 0
        bt = -1.0
        for j in range(ns):
            s = 0.0
            for k in range(K):
                s += Pk[k, j]
            if s > bt:
                bt = s
                bj = j
        total[ci] = bt
        bestT0[ci] = t_first + bj * step
        for k in range(K):
            parts[ci, k] = Pk[k, bj]
            flights[ci, k] = Fk[k, bj]
    return total, bestT0, parts, flights


@njit(parallel=True, cache=True, error_model="numpy")
def matched_scores(obs, T0, FL, LA, LF, LI, eoff, ioff, nseen, f0lo, f0hi, tk, P, V, start, shift_s, step, sigma, c):
    """Like full_scores, but each pass is tried only against the flight the candidate search matched it with
    (FL[candidate, pass], bank index or -1): a cheap check of many candidates before the full one."""
    n = len(T0)
    K = len(eoff) - 1
    ns = int(2 * shift_s / step) + 1
    total = np.zeros(n)
    bestT0 = np.zeros(n)
    for ci in prange(n):
        Pk = np.zeros((K, ns))
        t_first = T0[ci] - shift_s
        for k in range(K):
            f = FL[ci, k]
            if f < 0:
                continue
            a = start[f]
            b = start[f + 1]
            if b - a < 2:
                continue
            e0 = eoff[k]
            na = eoff[k + 1] - e0
            nu = na + ns - 1
            g = np.empty(nu)
            okg = np.zeros(nu, np.bool_)
            _curve_walk(tk, P, V, a, b, obs[ci], t_first + LA[e0], step, nu, c, g, okg)
            for j in range(ns):
                num = 0.0
                den = 0.0
                for u_ in range(ioff[k], ioff[k + 1]):
                    m = LI[u_]
                    u = m + j
                    if okg[u]:
                        num += LF[e0 + m] * g[u]
                        den += g[u] * g[u]
                if den == 0:
                    continue
                f0 = min(max(num / den, f0lo[k]), f0hi[k])
                tot = 0.0
                for u_ in range(ioff[k], ioff[k + 1]):
                    m = LI[u_]
                    u = m + j
                    if okg[u]:
                        x = (LF[e0 + m] - f0 * g[u]) / (2.0 * sigma)
                        x2 = x * x
                        if x2 < 1.0:
                            tot += (1.0 - x2) * (1.0 - x2)
                Pk[k, j] = tot / nseen[k]
        bj = 0
        bt = -1.0
        for j in range(ns):
            sm = 0.0
            for k in range(K):
                sm += Pk[k, j]
            if sm > bt:
                bt = sm
                bj = j
        total[ci] = bt
        bestT0[ci] = t_first + bj * step
    return total, bestT0



class FullScorer:
    """Inputs for full_scores: the passes' line grids (per time step) and time-sorted points in local metres."""

    def __init__(self, events, bank, pts, proj, dem, c, n_points=40, radius_m=12000.0, obs_height=1.5):
        self.events, self.bank, self.proj, self.dem, self.c = events, bank, proj, dem, c
        self.n_points, self.obs_height = n_points, obs_height
        self.win = np.array([e.window for e in events], np.float64)
        self.R = np.full(len(events), float(radius_m))
        p = pts[pts.fid.isin(bank.fids)]
        x, y = proj.xy(p.lat.to_numpy(), p.lon.to_numpy())
        o = np.argsort(p.t.to_numpy(), kind="stable")
        self.px, self.py = np.ascontiguousarray(x[o]), np.ascontiguousarray(y[o])
        self.pt = np.ascontiguousarray(p.t.to_numpy(np.float64)[o])
        self.pf = np.ascontiguousarray(bank.index(p.fid.to_numpy())[o]).astype(np.int64)
        self._g = {}

    def _grids(self, step, n_points=None):
        n_points = self.n_points if n_points is None else n_points
        key = (step, n_points)
        if key not in self._g:
            grids = [sc.line_grid(e, step) for e in self.events]
            eoff = np.r_[0, np.cumsum([len(x[0]) for x in grids])].astype(np.int64)
            LIs = []
            for x in grids:
                seen = np.flatnonzero(x[2])
                q = max(1, int(np.ceil(len(seen) / n_points)))
                LIs.append(seen[::q])
            ioff = np.r_[0, np.cumsum([len(v) for v in LIs])].astype(np.int64)
            self._g[key] = (np.concatenate([x[0] for x in grids]), np.concatenate([x[1] for x in grids]),
                             np.concatenate(LIs).astype(np.int64), eoff, ioff,
                             np.array([max(1, len(v)) for v in LIs], np.float64),
                             np.array([x[1][x[2]].min() * 0.85 - 0.03 * e.f0 for x, e in zip(grids, self.events)]),
                             np.array([x[1][x[2]].max() * 1.15 + 0.03 * e.f0 for x, e in zip(grids, self.events)]))
        return self._g[key]

    def __call__(self, x, y, T0, shift_s=8.0, step=2.0):
        """Scores at local-metre points x, y around start times T0. Returns total, T0, per-pass, flight ids."""
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        T0 = np.atleast_1d(np.asarray(T0, float))
        lat, lon = self.proj.latlon(x, y)
        obs = sc.observers(np.atleast_1d(lat), np.atleast_1d(lon), self.dem, self.obs_height)
        LA, LF, LI, eoff, ioff, nseen, f0lo, f0hi = self._grids(float(step))
        b = self.bank
        tot, T0b, parts, fl = full_scores(obs, np.ascontiguousarray(np.stack([x, y], 1)), T0, LA, LF, LI, eoff, ioff,
                                          nseen, f0lo, f0hi, self.win, self.R, b.t, b.pos, b.vel, b.start, self.px,
                                          self.py, self.pt, self.pf, float(shift_s), float(step), 1.5, self.c)
        fids = np.where(fl >= 0, b.fids[np.maximum(fl, 0)], -1)
        return tot, T0b, parts, fids

    def matched(self, x, y, T0, fids, shift_s=6.0, step=3.0, n_points=16):
        """Scores using only the flight matched to each pass (fids: candidates x passes, ADS-B flight ids or -1)."""
        x = np.atleast_1d(np.asarray(x, float))
        y = np.atleast_1d(np.asarray(y, float))
        T0 = np.atleast_1d(np.asarray(T0, float))
        # candidates sit on a lattice, so most share a place: look up the ground once per distinct place
        xy, inv = np.unique(np.round(np.stack([x, y], 1), 1), axis=0, return_inverse=True)
        lat, lon = self.proj.latlon(xy[:, 0], xy[:, 1])
        obs = np.ascontiguousarray(sc.observers(np.atleast_1d(lat), np.atleast_1d(lon), self.dem,
                                                self.obs_height)[inv.ravel()])
        LA, LF, LI, eoff, ioff, nseen, f0lo, f0hi = self._grids(float(step), n_points)
        b = self.bank
        fl = np.asarray(fids)
        idx = np.searchsorted(b.fids, np.maximum(fl, 0))
        idx = np.minimum(idx, len(b.fids) - 1)
        FL = np.where((fl >= 0) & (b.fids[idx] == fl), idx, -1).astype(np.int64)
        return matched_scores(obs, T0, np.ascontiguousarray(FL), LA, LF, LI, eoff, ioff, nseen, f0lo, f0hi, b.t, b.pos,
                              b.vel, b.start, float(shift_s), float(step), 1.5, self.c)
