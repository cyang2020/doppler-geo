"""Shared geometry for the search: local map projection, flight segments (with merging of nearly
straight runs), square blocks of the search area and their ground heights."""
from __future__ import annotations


import numpy as np
import pandas as pd
import pyproj
from numba import njit

from . import adsb
from .geo import geoid_undulation


# --------------------------------------------------------------------------- projection

class LocalProjection:
    """Lambert azimuthal equal-area centred on the search area (metres)."""

    def __init__(self, bbox):
        la = (bbox[0] + bbox[1]) / 2
        lo = (bbox[2] + bbox[3]) / 2
        crs = pyproj.CRS.from_proj4(f"+proj=laea +lat_0={la} +lon_0={lo} +datum=WGS84 +units=m +no_defs")
        self.fwd = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
        self.inv = pyproj.Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

    def xy(self, lat, lon):
        x, y = self.fwd.transform(np.asarray(lon), np.asarray(lat))
        return np.asarray(x), np.asarray(y)

    def latlon(self, x, y):
        lon, lat = self.inv.transform(np.asarray(x), np.asarray(y))
        return np.asarray(lat), np.asarray(lon)


# --------------------------------------------------------------------------- kernel


# --------------------------------------------------------------------------- segments

@njit(cache=True, error_model="numpy")
def _simplify(x, y, h, t, v, same, tol_xy, tol_z, tol_t, tol_v, max_pts):
    """Keep a subset of points so that every dropped point lies within the tolerances of the straight,
    constant-rate chord between the kept points around it (position, height, time, speed)."""
    n = len(x)
    keep = np.zeros(n, np.bool_)
    i = 0
    while i < n:
        keep[i] = True
        j = i + 1
        best = i + 1
        while j < n and same[j - 1] and j - i <= max_pts:
            dx, dy = x[j] - x[i], y[j] - y[i]
            L = np.sqrt(dx * dx + dy * dy)
            dt = t[j] - t[i]
            ok = dt > 0
            for m in range(i + 1, j):
                if not ok:
                    break
                ex, ey = x[m] - x[i], y[m] - y[i]
                if L > 1e-6:
                    along = (ex * dx + ey * dy) / L
                    perp = abs(ex * dy - ey * dx) / L
                    fr = along / L
                else:
                    perp = np.sqrt(ex * ex + ey * ey)
                    fr = (t[m] - t[i]) / dt
                if perp > tol_xy or fr < -0.02 or fr > 1.02:
                    ok = False
                    break
                fr_t = (t[m] - t[i]) / dt
                if abs(h[m] - (h[i] + fr_t * (h[j] - h[i]))) > tol_z:
                    ok = False
                    break
                if abs(t[m] - (t[i] + fr * dt)) > tol_t:
                    ok = False
                    break
                if abs(v[m] - (v[i] + fr * (v[j] - v[i]))) > tol_v:
                    ok = False
                    break
            if not ok:
                break
            best = j
            j += 1
        if best >= n or not (best < n and (best == i + 1 or True)):
            pass
        i = best if best > i else i + 1
    if n:
        keep[n - 1] = True
    return keep


class Segments:
    """Straight pieces between consecutive ADS-B points of the same flight.

    simplify_m > 0 merges runs of nearly collinear, steady points into one piece (every dropped point
    within simplify_m of the piece, and within 1.5 s / 3 m/s of its constant-rate timing and speed)."""

    def __init__(self, pts: pd.DataFrame, proj: LocalProjection, max_gap_s: float, simplify_m: float = 0.0):
        x, y = proj.xy(pts.lat.to_numpy(), pts.lon.to_numpy())
        h = adsb.fill_geometric_altitude(pts)
        # heights are ellipsoidal; the ground grid is converted to ellipsoidal too
        t = pts.t.to_numpy()
        fid = pts.fid.to_numpy()
        gs = pts.gs.to_numpy(np.float64)
        same = (fid[1:] == fid[:-1]) & (np.diff(t) <= max_gap_s) & (np.diff(t) > 0)
        self.n_raw = int(same.sum())
        if simplify_m > 0 and len(t) > 2:
            v = gs.copy()
            fd = np.r_[np.hypot(np.diff(x), np.diff(y)) / np.maximum(np.diff(t), 1e-3), 0.0]
            v = np.where(np.isfinite(v), v, fd)
            keep = _simplify(x, y, h, t.astype(np.float64), v, same, simplify_m, simplify_m, 1.5, 3.0, 200)
            brk = np.r_[0, np.cumsum(~same)]  # breaks before point i
            idx = np.flatnonzero(keep)
            same = (brk[idx[1:]] - brk[idx[:-1]]) == 0
            same &= fid[idx[1:]] == fid[idx[:-1]]
            x, y, h, t, fid, gs = x[idx], y[idx], h[idx], t[idx], fid[idx], gs[idx]
        a = np.flatnonzero(same)
        b = a + 1
        # fill missing ground speed from positions
        dist = np.hypot(x[b] - x[a], y[b] - y[a]) / (t[b] - t[a])
        gsa = np.where(np.isfinite(gs[a]), gs[a], dist)
        gsb = np.where(np.isfinite(gs[b]), gs[b], dist)
        self.fid = fid[a]
        self.ax, self.ay, self.az, self.at, self.av = x[a], y[a], h[a], t[a], gsa
        self.bx, self.by, self.bz, self.bt, self.bv = x[b], y[b], h[b], t[b], gsb
        prev_ok = np.r_[False, (a[1:] == b[:-1])]
        self.hasprev = prev_ok
        self.qx = np.r_[0.0, x[a][:-1]] if len(a) else np.zeros(0)
        self.qy = np.r_[0.0, y[a][:-1]] if len(a) else np.zeros(0)
        self.qz = np.r_[0.0, h[a][:-1]] if len(a) else np.zeros(0)
        self.n = len(a)

    def block_index(self, sel, block_m, pad):
        """(segment index, block key) pairs for segments in `sel` touching blocks within `pad`."""
        idx = np.flatnonzero(sel)
        lo_x = np.minimum(self.ax[idx], self.bx[idx]) - pad
        hi_x = np.maximum(self.ax[idx], self.bx[idx]) + pad
        lo_y = np.minimum(self.ay[idx], self.by[idx]) - pad
        hi_y = np.maximum(self.ay[idx], self.by[idx]) + pad
        bx0 = np.floor(lo_x / block_m).astype(np.int64)
        bx1 = np.floor(hi_x / block_m).astype(np.int64)
        by0 = np.floor(lo_y / block_m).astype(np.int64)
        by1 = np.floor(hi_y / block_m).astype(np.int64)
        out_s, out_b = [], []
        for ox in range(int((bx1 - bx0).max(initial=0)) + 1):
            for oy in range(int((by1 - by0).max(initial=0)) + 1):
                m = (bx0 + ox <= bx1) & (by0 + oy <= by1)
                out_s.append(idx[m])
                out_b.append(_bkey(bx0[m] + ox, by0[m] + oy))
        s = np.concatenate(out_s) if out_s else np.zeros(0, np.int64)
        b = np.concatenate(out_b) if out_b else np.zeros(0, np.int64)
        order = np.argsort(b, kind="stable")
        return s[order], b[order]

    def kernel_args(self):
        return (self.ax, self.ay, self.az, self.at, self.av, self.bx, self.by, self.bz, self.bt, self.bv,
                self.hasprev, self.qx, self.qy, self.qz)


def _bkey(bx, by):
    return (np.asarray(bx, np.int64) + (1 << 20)) * (1 << 21) + (np.asarray(by, np.int64) + (1 << 20))


def _bxy(key):
    return int(key // (1 << 21)) - (1 << 20), int(key % (1 << 21)) - (1 << 20)


def _groups(keys):
    """{block key: slice} for a sorted key array."""
    if len(keys) == 0:
        return {}
    u, start = np.unique(keys, return_index=True)
    end = np.r_[start[1:], len(keys)]
    return {int(k): slice(a, b) for k, a, b in zip(u, start, end)}


# --------------------------------------------------------------------------- driver


def _ground(bk, cfg, proj, dem, cache):
    if bk in cache:
        return cache[bk]
    nx = ny = int(round(cfg.block_m / cfg.grid_m))
    g = cfg.grid_m
    bxk, byk = _bxy(bk)
    x0, y0 = bxk * cfg.block_m, byk * cfg.block_m
    CX, CY = np.meshgrid(x0 + (np.arange(nx) + 0.5) * g, y0 + (np.arange(ny) + 0.5) * g)
    lat, lon = proj.latlon(CX, CY)
    la0, la1, lo0, lo1 = cfg.bbox
    mask = (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
    ground = np.zeros((ny, nx))
    if mask.any():
        N = float(geoid_undulation([lat[ny // 2, nx // 2]], [lon[ny // 2, nx // 2]])[0])
        elev = dem.elevation(np.where(mask, lat, lat[ny // 2, nx // 2]), np.where(mask, lon, lon[ny // 2, nx // 2]))
        ground = np.where(mask, elev, 0.0) + N + cfg.obs_height
    cache[bk] = (x0, y0, nx, ny, mask, ground)
    return cache[bk]


def _pad_bbox(b, deg):
    return (b[0] - deg, b[1] + deg, b[2] - deg, b[3] + deg)


