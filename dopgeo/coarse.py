"""Coarse search: from all flights of the day to a few thousand (place, start time) candidates.

For every detected pass, a shape table says which (speed, closest distance) pairs a straight pass
could have and at which closest-approach time. A compiled kernel walks every flight segment and
visits only the ground cells at an acceptable distance from it; each accepted fit gives an
interval of possible recording start times T0.

Level 1 (1 km cells, 10 s bins of T0): each fit marks its T0 interval for its pass; a (cell, bin)
scores the sum of the best fit of every pass marking it. The best 30 000 go on. Nothing is stored
per fit and nothing is sorted.

Level 2 (200 m cells inside those): the same kernel, now keeping each fit with its flight, but
only where the coarse cell and bin survived.

Level 3: each (cell, flight, pass) is re-scored with the flight's real trajectory: the exact
Doppler curve it produces at that cell, fitted to the pass's tracked line over the fit's T0
window. That gives a precise T0 interval per flight. A sweep over each cell's intervals finds
start times where several passes agree; those are the candidates.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from . import adsb, shape
from .geo import ecef
from .score import TrajectoryBank, curve_walk as _curve_walk, line_grid
from .dem import DEM
from .search import LocalProjection, Segments, _bkey, _bxy, _groups, _ground, _pad_bbox


@dataclass
class CoarseConfig:
    bbox: tuple
    t0_range: tuple
    audio_duration: float
    c: float = 343.0
    grid_m: float = 200.0  # level 2 cell size
    coarse_m: float = 1000.0  # level 1 cell size
    block_m: float = 40000.0
    bin_s: float = 10.0  # level 1 time bin for the recording start time
    obs_height: float = 1.5
    max_seg_gap_s: float = 60.0
    simplify_m: float = 25.0  # merge nearly straight runs of ADS-B points (0 = off)
    dem_zoom: int = 10
    min_votes: int = 3
    level1_keep: int = 30000  # (coarse cell, time bin) pairs passed to level 2, best by summed fit
    keep: int = 20000  # candidates handed to fine scoring
    real_vote: float = 0.6  # level 3: a pass votes if the real trajectory's curve fits this well (0..1)
    real_step: float = 3.0  # level 3 time step (s)
    real_stride: int = 1  # >1: level 3 first scans every `stride` time steps with a wider tolerance
    real_points: int = 24  # line points per pass used by the level-3 fits
    prevote_frac: float = 0.4  # straight-pass fits must have this share of passes overlapping to be re-scored
    real_pad: float = 4.0  # level 3 scans this much beyond each straight-pass T0 window (s)
    fill_s: float = 40.0  # a pass without a vote at a candidate still gets the nearest fit within this many s
    classes: tuple | None = None  # aircraft classes to consider (adsb.CLS_*); None = all
    probe: tuple | None = None  # (x, y, T0) in local metres / epoch: log how that place fares (debugging)
    log: callable = print


class _GroundCfg:
    def __init__(self, cfg, g):
        self.block_m, self.grid_m, self.bbox, self.obs_height = cfg.block_m, g, cfg.bbox, cfg.obs_height


# --------------------------------------------------------------------------- tables

def speed_bands(S: shape.ShapeTable, accept=None):
    """Accepted closest-distance range (m) at each speed of the table grid (0, 0 where none)."""
    ok = S.Q >= (S.accept if accept is None else accept)
    nv = ok.shape[0]
    lo = np.zeros(nv)
    hi = np.zeros(nv)
    step = S.lds[1] - S.lds[0]
    for i in range(nv):
        j = np.flatnonzero(ok[i])
        if len(j):
            lo[i] = 10 ** (S.lds[j[0]] - step)
            hi[i] = 10 ** (S.lds[j[-1]] + step)
    return lo, hi


class Tables:
    """All passes' shape tables stacked for compiled code (they share one speed/distance grid)."""

    def __init__(self, tables, blur_m, accept=None):
        T = [shape.blurred(S, blur_m) for S in tables]
        acc = float(T[0].accept if accept is None else accept)
        self.vs = T[0].vs
        self.lds = T[0].lds
        self.Q = np.ascontiguousarray(np.stack([t.Q for t in T]))
        self.TLO = np.ascontiguousarray(np.stack([t.TLO for t in T]))
        self.THI = np.ascontiguousarray(np.stack([t.THI for t in T]))
        b = [speed_bands(t, acc) for t in T]
        self.blo = np.ascontiguousarray(np.stack([x[0] for x in b]))
        self.bhi = np.ascontiguousarray(np.stack([x[1] for x in b]))
        self.accept = acc
        self.max_d = float(self.bhi.max())
        ok = self.Q >= self.accept
        self.vtlo = np.ascontiguousarray(np.where(ok, self.TLO, np.inf).min(axis=2))  # (K, nv)
        self.vthi = np.ascontiguousarray(np.where(ok, self.THI, -np.inf).max(axis=2))

    def args(self):
        return (self.vs, self.lds, self.Q, self.TLO, self.THI, np.stack([self.blo, self.bhi, self.vtlo, self.vthi]),
                self.accept)


# --------------------------------------------------------------------------- kernel pieces

@njit(cache=True, error_model="numpy")
def _interval(a, b, lo, hi):
    """x range where a*x + b is in [lo, hi]."""
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
def _lookup(vs, lds, Q, TLO, THI, k, v, d):
    """Fit score and accepted closest-approach time interval of pass k at speed v, distance d."""
    if d <= 0:
        return 0.0, 0.0, 0.0
    x = (v - vs[0]) / (vs[1] - vs[0])
    y = (np.log10(d) - lds[0]) / (lds[1] - lds[0])
    nv = Q.shape[1]
    nd = Q.shape[2]
    if x < 0 or y < 0 or x > nv - 1 or y > nd - 1:
        return 0.0, 0.0, 0.0
    i = min(int(x), nv - 2)
    j = min(int(y), nd - 2)
    fx = x - i
    fy = y - j
    q = (Q[k, i, j] * (1 - fx) * (1 - fy) + Q[k, i + 1, j] * fx * (1 - fy) + Q[k, i, j + 1] * (1 - fx) * fy
         + Q[k, i + 1, j + 1] * fx * fy)
    lo = 1e30
    hi = -1e30
    for a in range(2):
        for b in range(2):
            if Q[k, i + a, j + b] >= 0.5 * q:
                lo = min(lo, TLO[k, i + a, j + b])
                hi = max(hi, THI[k, i + a, j + b])
    if lo > hi:
        return 0.0, 0.0, 0.0
    return q, lo, hi


@njit(cache=True, error_model="numpy")
def _eval(i, ix, iy, x0, y0, g, ground, SG, vs, lds, Q, TLO, THI, accept, k, c, t0_lo, t0_hi):
    """Exact closest approach of segment i to cell (ix, iy): (ok, T0 lo, T0 hi, fit score)."""
    cx = x0 + (ix + 0.5) * g
    cy = y0 + (iy + 0.5) * g
    cz = ground[iy, ix]
    Ax, Ay, Az, At, Av = SG[i, 0], SG[i, 1], SG[i, 2], SG[i, 3], SG[i, 4]
    dx, dy, dz = SG[i, 5] - Ax, SG[i, 6] - Ay, SG[i, 7] - Az
    L2 = dx * dx + dy * dy + dz * dz
    if L2 < 1e-6:
        return False, 0.0, 0.0, 0.0  # stationary: no single closest-approach time
    s = ((cx - Ax) * dx + (cy - Ay) * dy + (cz - Az) * dz) / L2
    if s >= 1.0:
        return False, 0.0, 0.0, 0.0
    if s <= 0.0:
        # vertex A counts only if it is also past the end of the previous segment
        if SG[i, 10] == 0:
            return False, 0.0, 0.0, 0.0
        px, py, pz = SG[i, 11], SG[i, 12], SG[i, 13]
        ex, ey, ez = Ax - px, Ay - py, Az - pz
        if (cx - px) * ex + (cy - py) * ey + (cz - pz) * ez < ex * ex + ey * ey + ez * ez:
            return False, 0.0, 0.0, 0.0
        s = 0.0
    Px = Ax + s * dx - cx
    Py = Ay + s * dy - cy
    Pz = Az + s * dz - cz
    d = np.sqrt(Px * Px + Py * Py + Pz * Pz)
    v = Av + s * (SG[i, 9] - Av)
    q, tlo, thi = _lookup(vs, lds, Q, TLO, THI, k, v, d)
    if q < accept:
        return False, 0.0, 0.0, 0.0
    th = At + s * (SG[i, 8] - At) + d / c
    slack = 0.71 * g / max(v, 10.0) + 1.0
    a = th - thi - slack
    b = th - tlo + slack
    if b < t0_lo or a > t0_hi:
        return False, 0.0, 0.0, 0.0
    return True, a, b, q


@njit(cache=True, error_model="numpy")
def _act(mode, k, i, ix, iy, nx, T0a, T0b, q, t0_lo, bin_s, nb, mark, alive, r, nA, out, pos):
    """mode 0: mark time bins in mark[k]; mode 1: count kept fits; mode 2: write kept fits."""
    b0 = max(0, int((T0a - t0_lo) / bin_s))
    b1 = min(nb - 1, int((T0b - t0_lo) / bin_s))
    if mode == 0:
        c = iy * nx + ix
        qb = np.uint8(1 + int(q * 250))
        for b in range(b0, b1 + 1):
            if mark[k, c, b] < qb:
                mark[k, c, b] = qb  # concurrent writers only ever raise it; a lost race is harmless
        return pos
    p = (iy // r) * nA + (ix // r)
    lo = alive[p, b0 - 1] if b0 > 0 else 0
    if alive[p, b1] - lo <= 0:
        return pos
    if mode == 2:
        out[pos, 0] = iy * nx + ix
        out[pos, 1] = k
        out[pos, 2] = T0a
        out[pos, 3] = T0b
        out[pos, 4] = q
        out[pos, 5] = i
    return pos + 1


@njit(cache=True, error_model="numpy")
def _maybe(mode, alive, r, nA, ix, iy, sb0, sb1):
    """Levels 2: does the coarse parent of this cell have any alive bin in the segment's window?"""
    if mode == 0:
        return True
    p = (iy // r) * nA + (ix // r)
    lo = alive[p, sb0 - 1] if sb0 > 0 else 0
    return alive[p, sb1] - lo > 0


@njit(cache=True, error_model="numpy")
def _segment(mode, k, i, x0, y0, nx, ny, g, ground, gmin, gmax, mask, SG, vs, lds, Q, TLO, THI, BAND, accept,
             c, t0_lo, t0_hi, bin_s, nb, mark, alive, r, nA, out, pos):
    """Visit the cells near segment i that could be at an accepted closest distance.
    In modes 1/2 `alive` is the running count of alive bins per coarse cell (prefix sums)."""
    v0 = min(SG[i, 4], SG[i, 9])
    v1 = max(SG[i, 4], SG[i, 9])
    dv = vs[1] - vs[0]
    i0 = max(0, int(np.floor((v0 - vs[0]) / dv)))
    i1 = min(len(vs) - 1, int(np.ceil((v1 - vs[0]) / dv)))
    dlo = 1e30
    dhi = 0.0
    for m in range(i0, i1 + 1):
        if BAND[1, k, m] > 0:
            dlo = min(dlo, BAND[0, k, m])
            dhi = max(dhi, BAND[1, k, m])
    if dhi <= 0:
        return pos
    Ax, Ay = SG[i, 0], SG[i, 1]
    Bx, By = SG[i, 5], SG[i, 6]
    zlo = min(SG[i, 2], SG[i, 7])
    zhi = max(SG[i, 2], SG[i, 7])
    dzmin = max(0.0, max(zlo - gmax, gmin - zhi))
    dzmax = max(abs(zhi - gmin), abs(zlo - gmax))
    if dzmin >= dhi:
        return pos
    sb0 = 0
    sb1 = 0
    if mode != 0:
        tlo_k = 1e30
        thi_k = -1e30
        for m in range(i0, i1 + 1):
            tlo_k = min(tlo_k, BAND[2, k, m])
            thi_k = max(thi_k, BAND[3, k, m])
        if tlo_k > thi_k:
            return pos
        slack = 0.71 * g / max(v0, 10.0) + 1.0
        Ta = SG[i, 3] - thi_k - slack
        Tb = SG[i, 8] + dhi / c - tlo_k + slack
        if Tb < t0_lo or Ta > t0_hi:
            return pos
        sb0 = max(0, int((Ta - t0_lo) / bin_s))
        sb1 = min(nb - 1, int((Tb - t0_lo) / bin_s))
    hhi = np.sqrt(dhi * dhi - dzmin * dzmin)
    hlo = np.sqrt(max(0.0, dlo * dlo - dzmax * dzmax))
    pad = 0.3 * g + 5.0  # cells are tested at their centres; this only covers 2-D vs 3-D along-track
    dxh = Bx - Ax
    dyh = By - Ay
    Lh2 = dxh * dxh + dyh * dyh
    iy0 = max(0, int(np.floor((min(Ay, By) - hhi - pad - y0) / g)))
    iy1 = min(ny - 1, int(np.floor((max(Ay, By) + hhi + pad - y0) / g)))
    if iy0 > iy1:
        return pos
    if Lh2 < 1.0:
        ix0 = max(0, int(np.floor((min(Ax, Bx) - hhi - pad - x0) / g)))
        ix1 = min(nx - 1, int(np.floor((max(Ax, Bx) + hhi + pad - x0) / g)))
        for iy in range(iy0, iy1 + 1):
            for ix in range(ix0, ix1 + 1):
                if mask[iy, ix] and _maybe(mode, alive, r, nA, ix, iy, sb0, sb1):
                    ok, a, b, q = _eval(i, ix, iy, x0, y0, g, ground, SG, vs, lds, Q, TLO, THI, accept, k, c,
                                        t0_lo, t0_hi)
                    if ok:
                        pos = _act(mode, k, i, ix, iy, nx, a, b, q, t0_lo, bin_s, nb, mark, alive, r, nA, out, pos)
        return pos
    Lh = np.sqrt(Lh2)
    has_p = SG[i, 10] != 0
    pxv, pyv = SG[i, 11], SG[i, 12]
    exh = Ax - pxv
    eyh = Ay - pyv
    Lp2 = exh * exh + eyh * eyh
    two = hlo - pad > 0
    n_str = 2 if two else 1
    a_s = dxh / Lh2
    a_p = -dyh / Lh
    for iy in range(iy0, iy1 + 1):
        cy = y0 + (iy + 0.5) * g
        # along-track s and signed offset p are linear in the cell-centre x of this row
        b_s = (-Ax * dxh + (cy - Ay) * dyh) / Lh2
        b_p = (Ax * dyh + (cy - Ay) * dxh) / Lh
        s_lo, s_hi = _interval(a_s, b_s, -pad / Lh, 1.0 + pad / Lh)
        for st in range(n_str + 1):
            if st < n_str:
                if not two:
                    p_lo, p_hi = _interval(a_p, b_p, -hhi - pad, hhi + pad)
                elif st == 0:
                    p_lo, p_hi = _interval(a_p, b_p, hlo - pad, hhi + pad)
                else:
                    p_lo, p_hi = _interval(a_p, b_p, -hhi - pad, -(hlo - pad))
                xa = max(s_lo, p_lo)
                xb = min(s_hi, p_hi)
            else:
                # outside of a turn at vertex A: before this segment, past the previous one
                if not has_p or Lp2 < 1.0:
                    continue
                w_lo, w_hi = _interval(a_s, b_s, -(hhi + pad) / Lh, -pad / Lh)
                a_q = exh / Lp2
                b_q = (-pxv * exh + (cy - pyv) * eyh) / Lp2
                q_lo, q_hi = _interval(a_q, b_q, 1.0 - pad / np.sqrt(Lp2), 1e30)
                p_lo, p_hi = _interval(a_p, b_p, -hhi - pad, hhi + pad)
                xa = max(max(w_lo, q_lo), p_lo)
                xb = min(min(w_hi, q_hi), p_hi)
            if xa > xb:
                continue
            ix0 = max(0, int(np.ceil((xa - x0) / g - 0.5)))
            ix1 = min(nx - 1, int(np.floor((xb - x0) / g - 0.5)))
            for ix in range(ix0, ix1 + 1):
                if st == n_str:
                    cx = x0 + (ix + 0.5) * g
                    if a_s * cx + b_s >= -pad / Lh:
                        continue  # already visited by the main strip
                if mask[iy, ix] and _maybe(mode, alive, r, nA, ix, iy, sb0, sb1):
                    ok, a, b, q = _eval(i, ix, iy, x0, y0, g, ground, SG, vs, lds, Q, TLO, THI, accept, k, c,
                                        t0_lo, t0_hi)
                    if ok:
                        pos = _act(mode, k, i, ix, iy, nx, a, b, q, t0_lo, bin_s, nb, mark, alive, r, nA, out, pos)
    return pos


@njit(parallel=True, cache=True, error_model="numpy")
def mark_block(pk, ps, x0, y0, nx, ny, g, ground, gmin, gmax, mask, SG, vs, lds, Q, TLO, THI, BAND, accept,
               c, t0_lo, t0_hi, bin_s, nb, mark):
    """Level 1: mark[k, cell, bin] = best fit score (as 1..251) of pass k for that cell and start-time bin."""
    alive = np.zeros((1, 1), np.int32)
    out = np.zeros((1, 6))
    for j in prange(len(pk)):
        _segment(0, pk[j], ps[j], x0, y0, nx, ny, g, ground, gmin, gmax, mask, SG, vs, lds, Q, TLO, THI, BAND, accept,
                 c, t0_lo, t0_hi, bin_s, nb, mark, alive, 1, 1, out, 0)


@njit(parallel=True, cache=True, error_model="numpy")
def count_votes(mark, min_votes):
    """Per (cell, bin): number of passes marking it and the sum of their scores."""
    K, nc, nb = mark.shape
    cnt = np.zeros((nc, nb), np.uint8)
    tot = np.zeros((nc, nb), np.float32)
    for c in prange(nc):
        for b in range(nb):
            n = 0
            s = 0.0
            for k in range(K):
                m = mark[k, c, b]
                if m:
                    n += 1
                    s += (m - 1) / 250.0
            cnt[c, b] = n
            tot[c, b] = s
    return cnt, tot


@njit(cache=True, error_model="numpy")
def select_top(cnt, tot, min_votes, keep):
    """(cell, bin, votes, rank) of the best `keep` entries with >= min_votes passes, rank = summed fit."""
    nc, nb = cnt.shape
    nbk = 4096
    hist = np.zeros(nbk, np.int64)
    top = 0.0
    for c in range(nc):
        for b in range(nb):
            if cnt[c, b] >= min_votes and tot[c, b] > top:
                top = tot[c, b]
    if top <= 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    scale = (nbk - 1) / top
    for c in range(nc):
        for b in range(nb):
            if cnt[c, b] >= min_votes:
                hist[int(tot[c, b] * scale)] += 1
    lvl = nbk - 1
    acc = 0
    while lvl > 0:
        if acc + hist[lvl] > keep and acc > 0:
            break
        acc += hist[lvl]
        lvl -= 1
    lvl += 1
    n = 0
    for v in range(lvl, nbk):
        n += hist[v]
    cc = np.empty(n, np.int64)
    bb = np.empty(n, np.int64)
    vv = np.empty(n, np.int64)
    rk = np.empty(n)
    m = 0
    for c in range(nc):
        for b in range(nb):
            if cnt[c, b] >= min_votes and int(tot[c, b] * scale) >= lvl and m < n:
                cc[m] = c
                bb[m] = b
                vv[m] = cnt[c, b]
                rk[m] = tot[c, b]
                m += 1
    o = np.argsort(-rk[:m])[:keep]
    return cc[o], bb[o], vv[o], rk[o]




@njit(cache=True, error_model="numpy")
def prevote(cell, ev, lo, hi, K, min_votes):
    """Rows that are ever active while >= min_votes different passes overlap in the same cell."""
    n = len(cell)
    keep = np.zeros(n, np.bool_)
    order = np.argsort(cell, kind="mergesort")
    pts = np.empty(2 * n)
    act = np.empty(n, np.int64)
    posv = np.empty(n, np.int64)
    cnt = np.zeros(K, np.int64)
    a = 0
    while a < n:
        b = a
        c0 = cell[order[a]]
        while b < n and cell[order[b]] == c0:
            b += 1
        m = b - a
        for kk in range(K):
            cnt[kk] = 0
        nd = 0
        for u in range(a, b):
            kk = ev[order[u]]
            if cnt[kk] == 0:
                nd += 1
            cnt[kk] = 1
        if nd >= min_votes:
            for u in range(m):
                r = order[a + u]
                pts[2 * u] = lo[r] - 1e-6
                pts[2 * u + 1] = hi[r]
            ordp = np.argsort(pts[: 2 * m])
            for kk in range(K):
                cnt[kk] = 0
            na = 0
            nd = 0
            for o in range(2 * m):
                e = ordp[o]
                u = e // 2
                r = order[a + u]
                kk = ev[r]
                if e % 2 == 0:
                    act[na] = u
                    posv[u] = na
                    na += 1
                    if cnt[kk] == 0:
                        nd += 1
                        if nd == min_votes:
                            for w in range(na):
                                keep[order[a + act[w]]] = True
                    cnt[kk] += 1
                    if nd >= min_votes:
                        keep[r] = True
                else:
                    p_ = posv[u]
                    na -= 1
                    last = act[na]
                    act[p_] = last
                    posv[last] = p_
                    cnt[kk] -= 1
                    if cnt[kk] == 0:
                        nd -= 1
        a = b
    return keep


@njit(cache=True, error_model="numpy")
def merge_rows(cell, fid, k, lo, hi, order, gap):
    """Rows with the same (cell, flight, pass) and overlapping start-time windows become one."""
    n = len(order)
    oc = np.empty(n, np.int64)
    of = np.empty(n, np.int64)
    ok_ = np.empty(n, np.int64)
    ol = np.empty(n)
    oh = np.empty(n)
    m = -1
    for t in range(n):
        i = order[t]
        if m >= 0 and oc[m] == cell[i] and of[m] == fid[i] and ok_[m] == k[i] and lo[i] <= oh[m] + gap:
            oh[m] = max(oh[m], hi[i])
            ol[m] = min(ol[m], lo[i])
        else:
            m += 1
            oc[m] = cell[i]
            of[m] = fid[i]
            ok_[m] = k[i]
            ol[m] = lo[i]
            oh[m] = hi[i]
    return oc[: m + 1], of[: m + 1], ok_[: m + 1], ol[: m + 1], oh[: m + 1]


@njit(cache=True, error_model="numpy")
def _shift_score(LF, LI, i0, i1, g, okg, j, f0lo, f0hi, h, nseen):
    """Score of one start-time shift j: closed-form rest frequency, then a smooth bump per line point."""
    num = 0.0
    den = 0.0
    for q in range(i0, i1):
        m = LI[q]
        u = m + j
        if okg[u]:
            num += LF[m] * g[u]
            den += g[u] * g[u]
    if den == 0:
        return 0.0
    f0 = min(max(num / den, f0lo), f0hi)
    tot = 0.0
    for q in range(i0, i1):
        m = LI[q]
        u = m + j
        if okg[u]:
            x = (LF[m] - f0 * g[u]) / h
            x2 = x * x
            if x2 < 1.0:
                tot += (1.0 - x2) * (1.0 - x2)
    return tot / nseen


@njit(parallel=True, cache=True, error_model="numpy")
def profile_rows(obs, ro, rt, rk, rlo, rhi, tk, P, V, start, LA, LF, LI, eoff, ioff, nseen, f0lo, f0hi, step, sigma,
                 c, max_ns, drop, stride):
    """Level 3: fit of the real trajectory's Doppler curve to the pass's line, over the row's start-time window.
    LI lists the line points used (a thinned subset). Shifts are scanned every 3 steps with a wide bump, then
    every step around the best. Returns best score, its T0, and the T0 interval within `drop` of the best."""
    n = len(ro)
    best = np.zeros(n)
    bT0 = np.zeros(n)
    ilo = np.zeros(n)
    ihi = np.zeros(n)
    for r in prange(n):
        a = start[rt[r]]
        b = start[rt[r] + 1]
        if b - a < 2:
            continue
        k = rk[r]
        e0 = eoff[k]
        na = eoff[k + 1] - e0
        ns = min(max_ns, int((rhi[r] - rlo[r]) / step) + 1)
        nu = na + ns - 1
        g = np.empty(nu)
        okg = np.zeros(nu, np.bool_)
        _curve_walk(tk, P, V, a, b, obs[ro[r]], rlo[r] + LA[e0], step, nu, c, g, okg)
        # LI holds absolute indices into LA/LF; make them relative to this pass for u = m + j
        i0 = ioff[k]
        i1 = ioff[k + 1]
        prof = np.full(ns, -1.0)
        coarse_best = -1.0
        cj = 0
        for j in range(0, ns, stride):
            v = _shift_score(LF[e0:], LI, i0, i1, g, okg, j, f0lo[k], f0hi[k], max(2.0, step) * stride * sigma,
                             nseen[k])
            if v > coarse_best:
                coarse_best = v
                cj = j
        for j in range(max(0, cj - stride - 1), min(ns, cj + stride + 2)):
            prof[j] = _shift_score(LF[e0:], LI, i0, i1, g, okg, j, f0lo[k], f0hi[k], max(2.0, step) * sigma, nseen[k])
        jb = max(0, cj - stride - 1)
        for j in range(max(0, cj - stride - 1), min(ns, cj + stride + 2)):
            if prof[j] > prof[jb]:
                jb = j
        best[r] = prof[jb]
        bT0[r] = rlo[r] + jb * step
        thr = prof[jb] - drop
        j0 = jb
        while j0 > 0 and prof[j0 - 1] >= thr:
            j0 -= 1
        j1 = jb
        while j1 < ns - 1 and prof[j1 + 1] >= thr:
            j1 += 1
        ilo[r] = rlo[r] + j0 * step - 0.5 * step
        ihi[r] = rlo[r] + j1 * step + 0.5 * step
    return best, bT0, ilo, ihi


@njit(parallel=True, cache=True, error_model="numpy")
def fits_block(pk, ps, x0, y0, nx, ny, g, ground, gmin, gmax, mask, SG, vs, lds, Q, TLO, THI, BAND, accept,
               c, t0_lo, t0_hi, bin_s, nb, alive, r, nA):
    """Level 2: fits whose coarse cell / time bin is alive. Rows: cell, pass, T0 lo, T0 hi, score, segment."""
    n = len(pk)
    mark = np.zeros((1, 1, 1), np.uint8)
    dummy = np.zeros((1, 6))
    counts = np.zeros(n + 1, np.int64)
    for j in prange(n):
        counts[j + 1] = _segment(1, pk[j], ps[j], x0, y0, nx, ny, g, ground, gmin, gmax, mask, SG, vs, lds, Q, TLO,
                                 THI, BAND, accept, c, t0_lo, t0_hi, bin_s, nb, mark, alive, r, nA, dummy, 0)
    offs = np.cumsum(counts)
    out = np.empty((offs[-1], 6))
    for j in prange(n):
        _segment(2, pk[j], ps[j], x0, y0, nx, ny, g, ground, gmin, gmax, mask, SG, vs, lds, Q, TLO, THI, BAND,
                 accept, c, t0_lo, t0_hi, bin_s, nb, mark, alive, r, nA, out, offs[j])
    return out


# --------------------------------------------------------------------------- level 2 voting

@njit(cache=True, error_model="numpy")
def vote(cell, ev, lo, hi, q, K, min_votes, vq, fill_s):
    """Per cell, sweep the passes' T0 intervals. Only fits scoring >= vq count as votes. At every local
    maximum of agreement with >= min_votes passes, emit for each pass its best active fit, or, if it
    has none, its best fit of that cell whose interval lies within fill_s of the agreement window."""
    n = len(cell)
    order = np.argsort(cell, kind="mergesort")
    cap = max(16, n)
    out_rows = np.full((cap, K), -1, np.int64)
    out_T0 = np.zeros(cap)
    out_votes = np.zeros(cap, np.int64)
    out_score = np.zeros(cap)
    n_out = 0
    pts = np.empty(2 * n)
    act = np.empty(n, np.int64)
    posv = np.empty(n, np.int64)
    cnt = np.zeros(K, np.int64)
    best = np.empty(K, np.int64)
    bq = np.empty(K)
    a = 0
    while a < n:
        b = a
        c0 = cell[order[a]]
        while b < n and cell[order[b]] == c0:
            b += 1
        m = b - a
        nd = 0
        for kk in range(K):
            cnt[kk] = 0
        for u in range(a, b):
            r = order[u]
            if q[r] >= vq:
                kk = ev[r]
                if cnt[kk] == 0:
                    nd += 1
                cnt[kk] = 1
        if nd >= min_votes:
            for u in range(m):
                r = order[a + u]
                pts[2 * u] = lo[r] - 1e-6  # starts sort before ends at equal times
                pts[2 * u + 1] = hi[r]
            ordp = np.argsort(pts[: 2 * m])
            for kk in range(K):
                cnt[kk] = 0
            na = 0
            nd = 0
            last_start = False
            for o in range(2 * m):
                e = ordp[o]
                u = e // 2
                r = order[a + u]
                kk = ev[r]
                strong = q[r] >= vq
                if e % 2 == 0:
                    act[na] = u
                    posv[u] = na
                    na += 1
                    if strong:
                        if cnt[kk] == 0:
                            nd += 1
                        cnt[kk] += 1
                        last_start = True
                    continue
                if last_start and nd >= min_votes and n_out < cap:
                    for k2 in range(K):
                        best[k2] = -1
                        bq[k2] = 0.0
                    for w in range(na):
                        rb = order[a + act[w]]
                        if q[rb] < vq:
                            continue
                        kb = ev[rb]
                        if q[rb] > bq[kb]:
                            bq[kb] = q[rb]
                            best[kb] = rb
                    tlo = -1e30
                    thi = 1e30
                    sc = 0.0
                    for k2 in range(K):
                        if best[k2] >= 0:
                            tlo = max(tlo, lo[best[k2]])
                            thi = min(thi, hi[best[k2]])
                            sc += bq[k2]
                    if tlo > thi:
                        tlo = thi = hi[r]
                    # passes without a vote here: nearest acceptable fit of the same cell
                    for k2 in range(K):
                        if best[k2] >= 0:
                            continue
                        bgap = fill_s
                        for w in range(m):
                            rw = order[a + w]
                            if ev[rw] != k2:
                                continue
                            gap = max(0.0, max(lo[rw] - thi, tlo - hi[rw]))
                            if gap <= bgap and (best[k2] < 0 or gap < bgap or q[rw] > bq[k2]):
                                bgap = gap
                                bq[k2] = q[rw]
                                best[k2] = rw
                    for k2 in range(K):
                        out_rows[n_out, k2] = best[k2]
                    out_T0[n_out] = 0.5 * (tlo + thi)
                    out_votes[n_out] = nd
                    out_score[n_out] = sc
                    n_out += 1
                last_start = False
                p_ = posv[u]
                na -= 1
                last = act[na]
                act[p_] = last
                posv[last] = p_
                if strong:
                    cnt[kk] -= 1
                    if cnt[kk] == 0:
                        nd -= 1
        a = b
    return out_rows[:n_out], out_T0[:n_out], out_votes[:n_out], out_score[:n_out]


# --------------------------------------------------------------------------- driver

def _seg_table(segs):
    return np.ascontiguousarray(np.stack([segs.ax, segs.ay, segs.az, segs.at, segs.av, segs.bx, segs.by, segs.bz,
                                          segs.bt, segs.bv, segs.hasprev.astype(np.float64), segs.qx, segs.qy,
                                          segs.qz], axis=1))


def _pairs(ks, sidx):
    """(pass, segment) work items for passes ks, interleaved so threads get mixed work."""
    ks = np.asarray(ks, np.int64)
    pk = np.tile(ks, len(sidx))
    ps = np.repeat(sidx.astype(np.int64), len(ks))
    return pk, ps


def _pairs_in_time(ks, sidx, SG, bins, t0_lo, bin_s, tmin, tmax, travel):
    """(pass, segment) items whose segment time can reach an alive start-time bin for that pass."""
    T0a = t0_lo + bins * bin_s
    T0b = T0a + bin_s
    # merge the bins into intervals
    starts = [T0a[0]]
    ends = [T0b[0]]
    for a, b in zip(T0a[1:], T0b[1:]):
        if a <= ends[-1] + 1e-6:
            ends[-1] = b
        else:
            starts.append(a)
            ends.append(b)
    starts = np.array(starts)
    ends = np.array(ends)
    seg_a = SG[sidx, 3]
    seg_b = SG[sidx, 8]
    pk, ps = [], []
    for k in ks:
        # heard time = segment time + travel (0..travel); T0 = heard - closest-approach audio time
        lo = starts + tmin[k] - travel
        hi = ends + tmax[k]
        j = np.searchsorted(hi, seg_a)  # first interval ending after the segment starts
        ok = (j < len(lo)) & (lo[np.minimum(j, len(lo) - 1)] <= seg_b)
        sel = sidx[ok]
        pk.append(np.full(len(sel), k, np.int64))
        ps.append(sel.astype(np.int64))
    if not pk:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    pk = np.concatenate(pk)
    ps = np.concatenate(ps)
    order = np.argsort(ps, kind="stable")
    return pk[order], ps[order]


def run(events, tables, day_dirs, cfg: CoarseConfig):
    """Return (candidates dict, flights, points, segments, projection)."""
    log = cfg.log
    T = time.time()
    proj = LocalProjection(cfg.bbox)
    K = len(events)
    t_lo = cfg.t0_range[0] - 900
    t_hi = cfg.t0_range[1] + cfg.audio_duration + 900
    fl, pts = adsb.load(day_dirs, bbox=_pad_bbox(cfg.bbox, 0.25), t_range=(t_lo, t_hi), classes=cfg.classes)
    if pts.empty:
        raise RuntimeError("no ADS-B points in the search window/area")
    segs = Segments(pts, proj, cfg.max_seg_gap_s, cfg.simplify_m)
    SG = _seg_table(segs)
    log(f"loaded {len(fl)} flights / {len(pts)} points; {segs.n_raw} segments merged into {segs.n} "
        f"({time.time() - T:.1f}s)")
    dem = DEM(cfg.dem_zoom)
    min_votes = min(cfg.min_votes, K)
    t0_lo, t0_hi = map(float, cfg.t0_range)
    nb = int(np.ceil((t0_hi - t0_lo) / cfg.bin_s)) + 1
    bank = TrajectoryBank(pts, fl.fid.to_numpy())

    # ---- level 1: coarse cells, time bins, summed fits
    gA = cfg.coarse_m
    TA = Tables(tables, 0.71 * gA)
    s_idx, b_key = segs.block_index(np.ones(segs.n, bool), cfg.block_m, TA.max_d + gA)
    groups = _groups(b_key)
    nA = int(round(cfg.block_m / gA))
    mark = np.zeros((K, nA * nA, nb), np.uint8)
    cacheA = {}
    lvl1 = {}
    t1 = time.time()
    for bk in sorted(groups):
        x0, y0, nx, ny, mask, ground = _ground(bk, _GroundCfg(cfg, gA), proj, dem, cacheA)
        if not mask.any():
            continue
        mark[:] = 0
        gmin, gmax = float(ground[mask].min()), float(ground[mask].max())
        pk, ps = _pairs(np.arange(K), s_idx[groups[bk]])
        mark_block(pk, ps, x0, y0, nx, ny, gA, ground, gmin, gmax, mask, SG, *TA.args(), cfg.c, t0_lo, t0_hi,
                   cfg.bin_s, nb, mark)
        cnt, tot = count_votes(mark, min_votes)
        if cfg.probe is not None:
            px_, py_, pT_ = cfg.probe
            if bk == int(_bkey(int(np.floor(px_ / cfg.block_m)), int(np.floor(py_ / cfg.block_m)))):
                pc_ = int((py_ - y0) // gA) * nA + int((px_ - x0) // gA)
                pb_ = int((pT_ - t0_lo) // cfg.bin_s)
                log(f"  probe level 1 truth cell: per pass best q {[round((int(m) - 1) / 250, 2) for m in mark[:, pc_, pb_]]}, "
                    f"sum {tot[pc_, pb_]:.2f}; entries in block with a higher sum {int((tot > tot[pc_, pb_]).sum())}")
        cc, bb, vv, rk = select_top(cnt, tot, min_votes, cfg.level1_keep)
        if len(cc):
            lvl1[bk] = (cc, bb, vv, rk)
    if not lvl1:
        log("level 1: no place explains enough passes")
        return None, fl, pts, segs, proj
    allrank = np.concatenate([v[3] for v in lvl1.values()])
    cut = np.sort(allrank)[::-1][min(cfg.level1_keep, len(allrank)) - 1]
    vmax = max(int(v[2].max()) for v in lvl1.values())
    n_alive = int((allrank >= cut).sum())
    if cfg.probe is not None:
        px, py, pT = cfg.probe
        pbk = int(_bkey(int(np.floor(px / cfg.block_m)), int(np.floor(py / cfg.block_m))))
        pc = int((py - _bxy(pbk)[1] * cfg.block_m) // gA) * nA + int((px - _bxy(pbk)[0] * cfg.block_m) // gA)
        pb = int((pT - t0_lo) // cfg.bin_s)
        if pbk in lvl1:
            cc_, bb_, vv_, rk_ = lvl1[pbk]
            sel = (cc_ == pc) & (np.abs(bb_ - pb) <= 4)
            log(f"  probe level 1: block kept, cell/bins near truth kept: {int(sel.sum())}, their scores "
                f"{np.round(rk_[sel], 2).tolist()[:6]}, cut {cut:.2f}, best {allrank.max():.2f}")
        else:
            log("  probe level 1: block has no kept entries")
    log(f"  level 1 ({gA:.0f} m cells, {cfg.bin_s:.0f} s bins): {time.time() - t1:.1f}s; up to {vmax} of {K} passes "
        f"agree; {n_alive} (cell, time) kept")

    # ---- level 2: fine cells, straight-pass fits with their flights
    # ---- level 3: those fits re-scored with each flight's real trajectory, then voted on precise start times
    g = cfg.grid_m
    TB = Tables(tables, 0.71 * g)
    r = int(round(gA / g))
    cacheB = {}
    step = cfg.real_step
    grids = [line_grid(e, step) for e in events]
    eoff = np.r_[0, np.cumsum([len(x[0]) for x in grids])].astype(np.int64)
    LA = np.concatenate([x[0] for x in grids])
    LF = np.concatenate([x[1] for x in grids])
    LIs = []
    for x in grids:
        seen = np.flatnonzero(x[2])
        q = max(1, int(np.ceil(len(seen) / cfg.real_points)))
        LIs.append(seen[::q])
    ioff = np.r_[0, np.cumsum([len(v) for v in LIs])].astype(np.int64)
    LI = np.concatenate(LIs).astype(np.int64)
    nseen = np.array([max(1, len(v)) for v in LIs], np.float64)
    f0lo = np.array([x[1][x[2]].min() * 0.85 - 0.03 * e.f0 for x, e in zip(grids, events)])
    f0hi = np.array([x[1][x[2]].max() * 1.15 + 0.03 * e.f0 for x, e in zip(grids, events)])
    res = dict(x=[], y=[], fid=[], T0=[], votes=[], coarse=[])
    n_fits = n_rows = 0
    t2 = time.time()
    tk = tp = tv = 0.0
    tmin = np.array([TB.TLO[k][TB.Q[k] >= TB.accept].min() for k in range(K)])
    tmax = np.array([TB.THI[k][TB.Q[k] >= TB.accept].max() for k in range(K)])
    for bk, (cc, bb, nn, rk) in lvl1.items():
        keep = rk >= cut
        if not keep.any():
            continue
        alive = np.zeros((nA * nA, nb), np.int32)
        alive[cc[keep], bb[keep]] = 1
        alive = np.cumsum(alive, axis=1, dtype=np.int32)
        pmask = np.zeros(nA * nA, bool)
        pmask[cc[keep]] = True
        x0, y0, nx, ny, mask, ground = _ground(bk, _GroundCfg(cfg, g), proj, dem, cacheB)
        mask = mask & np.kron(pmask.reshape(nA, nA), np.ones((r, r), bool))[:ny, :nx]
        if not mask.any():
            continue
        tq = time.time()
        gmin, gmax = float(ground[mask].min()), float(ground[mask].max())
        pk, ps = _pairs_in_time(np.arange(K), s_idx[groups[bk]], SG, np.unique(bb[keep]), t0_lo, cfg.bin_s, tmin,
                                tmax, TB.max_d / cfg.c + 10)
        if len(pk) == 0:
            continue
        H = fits_block(pk, ps, x0, y0, nx, ny, g, ground, gmin, gmax, mask, SG, *TB.args(), cfg.c, t0_lo, t0_hi,
                       cfg.bin_s, nb, alive, r, nA)
        n_fits += len(H)
        tk += time.time() - tq
        if len(H) == 0:
            continue
        # level 3
        tq = time.time()
        cell = H[:, 0].astype(np.int64)
        kk = H[:, 1].astype(np.int64)
        pre = prevote(cell, kk, H[:, 2], H[:, 3], K, max(min_votes, int(np.ceil(cfg.prevote_frac * K))))
        H, cell, kk = H[pre], cell[pre], kk[pre]
        if len(H) == 0:
            continue
        fid = segs.fid[H[:, 5].astype(np.int64)]
        order = np.lexsort((H[:, 2], kk, fid, cell))
        mc, mf, mk, ml, mh = merge_rows(cell, fid, kk, H[:, 2] - cfg.real_pad, H[:, 3] + cfg.real_pad, order, 4.0)
        n_rows += len(mc)
        ucell, ro = np.unique(mc, return_inverse=True)
        iy, ix = np.divmod(ucell, nx)
        cx = x0 + (ix + 0.5) * g
        cy = y0 + (iy + 0.5) * g
        la, lo_ = proj.latlon(cx, cy)
        obs = ecef(la, lo_, ground[iy, ix])
        rt = bank.index(mf)
        # flight-major order keeps each trajectory in cache while its cells are scored
        fo = np.argsort(rt, kind="stable")
        best, bT0, ilo, ihi = (np.empty(len(fo)) for _ in range(4))
        b_, t_, l_, h_ = profile_rows(obs, ro[fo].astype(np.int64), rt[fo], mk[fo], ml[fo], mh[fo], bank.t, bank.pos,
                                      bank.vel, bank.start, LA, LF, LI, eoff, ioff, nseen, f0lo, f0hi, step, 1.5, cfg.c, 80,
                                      0.1, cfg.real_stride)
        best[fo], bT0[fo], ilo[fo], ihi[fo] = b_, t_, l_, h_
        tp += time.time() - tq
        if cfg.probe is not None and bk == int(_bkey(int(np.floor(cfg.probe[0] / cfg.block_m)),
                                                      int(np.floor(cfg.probe[1] / cfg.block_m)))):
            pc = int((cfg.probe[1] - y0) // g) * nx + int((cfg.probe[0] - x0) // g)
            hc = np.flatnonzero((mc == pc) & (ml < cfg.probe[2] + 20) & (mh > cfg.probe[2] - 20))
            for q_ in hc[:20]:
                log(f"    truth-cell row: pass {mk[q_]} fid {mf[q_]} window {ml[q_] - cfg.probe[2]:+.0f}..{mh[q_] - cfg.probe[2]:+.0f}"
                    f" -> real fit {best[q_]:.2f} at {bT0[q_] - cfg.probe[2]:+.1f}s, interval {ilo[q_] - cfg.probe[2]:+.1f}..{ihi[q_] - cfg.probe[2]:+.1f}")
        tq = time.time()
        ok = best >= 0.3
        if not ok.any():
            continue
        rows, T0, votes, sc = vote(mc[ok], mk[ok], ilo[ok], ihi[ok], best[ok], K, min_votes, cfg.real_vote, 10.0)
        tv += time.time() - tq
        if len(rows) == 0:
            continue
        idx = np.flatnonzero(ok)
        anyrow = idx[rows.max(axis=1)]
        ci = ro[anyrow]
        res["x"].append(cx[ci])
        res["y"].append(cy[ci])
        res["fid"].append(np.where(rows >= 0, mf[idx[np.maximum(rows, 0)]], -1))
        res["T0"].append(T0)
        res["votes"].append(votes)
        res["coarse"].append(sc)
    if not res["T0"]:
        log("level 3: nothing left")
        return None, fl, pts, segs, proj
    res = {k: np.concatenate(v) for k, v in res.items()}
    order = np.argsort(-res["coarse"], kind="stable")
    if cfg.probe is not None:
        near = (np.hypot(res["x"] - cfg.probe[0], res["y"] - cfg.probe[1]) < 300) & (np.abs(res["T0"] - cfg.probe[2]) < 30)
        rank = np.empty(len(order), np.int64)
        rank[order] = np.arange(len(order))
        log(f"  probe: {int(near.sum())} candidates near it; best ranks {np.sort(rank[near])[:5].tolist()} of {len(rank)}; "
            f"votes {res['votes'][near][:5].tolist()} scores {np.round(res['coarse'][near][:5], 2).tolist()}; "
            f"top scores {np.round(res['coarse'][order[:5]], 2).tolist()}")
    order = order[: cfg.keep]
    res = {k: v[order] for k, v in res.items()}
    vc = np.bincount(res["votes"], minlength=K + 1)
    log(f"  levels 2+3 ({g:.0f} m cells): {time.time() - t2:.1f}s (straight fits {tk:.1f}s, real-curve fits {tp:.1f}s, "
        f"voting {tv:.1f}s); {n_fits} fits -> {n_rows} flight/cell/pass rows; {len(res['T0'])} candidates kept, by votes "
        f"{dict((v, int(vc[v])) for v in range(K + 1) if vc[v])} (total {time.time() - T:.1f}s)")
    cands = dict(x=res["x"], y=res["y"], T0=res["T0"], votes=res["votes"], coarse=res["coarse"], fid=res["fid"],
                 bank=bank)
    return cands, fl, pts, segs, proj
