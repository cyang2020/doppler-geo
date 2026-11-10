"""Measure one aircraft pass in the audio: follow its tone line and fit a straight-pass Doppler
curve, giving the heard closest-approach time, rest frequency, speed and closest slant distance,
each with a range that the search will treat as "anything in here is acceptable"."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from numba import njit
from scipy.optimize import least_squares

from .audio import Spectrogram
from .physics import straight_line


@dataclass
class Event:
    t_heard: float  # audio time (s) when the closest-approach sound arrives
    f0: float  # rest frequency of the tracked line (Hz)
    speed: float  # fitted speed (m/s)
    dist: float  # fitted closest slant distance (m)
    t_range: tuple  # acceptable t_heard range
    f0_range: tuple
    speed_range: tuple
    dist_range: tuple
    window: tuple  # audio-time span over which the line is visible (used for scoring)
    band: tuple  # frequency band the line occupies
    harmonics: list = field(default_factory=lambda: [1.0])  # extra lines to score, as multiples of f0
    rms_hz: float = 0.0
    line_t: list = field(default_factory=list)  # the tracked tone line (audio s, Hz), ~0.5 s spacing
    line_f: list = field(default_factory=list)
    quality: float = 0.0  # visible seconds x mean line strength; higher = clearer pass
    confident: bool = True  # False = looks like a pass but too short/ambiguous to search on
    note: str = ""

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        for k in ("t_range", "f0_range", "speed_range", "dist_range", "window", "band"):
            d[k] = tuple(d[k])
        return cls(**d)


@njit(cache=True)
def _ridge_dp(S, max_jump, penalty):
    F, T = S.shape
    acc = np.empty((F, T))
    back = np.zeros((F, T), np.int64)
    acc[:, 0] = S[:, 0]
    for t in range(1, T):
        for f in range(F):
            best = -1e18
            bi = f
            for g in range(max(0, f - max_jump), min(F, f + max_jump + 1)):
                v = acc[g, t - 1] - penalty * abs(f - g)
                if v > best:
                    best = v
                    bi = g
            acc[f, t] = best + S[f, t]
            back[f, t] = bi
    path = np.empty(T, np.int64)
    path[T - 1] = np.argmax(acc[:, T - 1])
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[path[t], t]
    return path


def track_line(spec: Spectrogram, t_lo, t_hi, f_lo, f_hi, max_slope_hz_s=6.0, penalty=0.15, min_strength=1.0):
    """Follow the strongest continuous tone inside the box. Returns times, freqs, strengths of
    the frames where the line is actually visible."""
    sub = spec.crop(t_lo, t_hi, f_lo, f_hi)
    jump = max(1, int(np.ceil(max_slope_hz_s * sub.dt / sub.df)))
    S = sub.values.astype(np.float64)
    path = _ridge_dp(S, jump, penalty)
    t = sub.times
    # sub-bin peak refinement
    f = np.empty(len(path))
    s = np.empty(len(path))
    for i, k in enumerate(path):
        k0, k1 = max(k - 1, 0), min(k + 1, S.shape[0] - 1)
        a, b, c = S[k0, i], S[k, i], S[k1, i]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if k0 < k < k1 and den < 0 else 0.0
        f[i] = sub.f0 + (k + off) * sub.df
        s[i] = b
    # visible = smoothed strength above threshold
    sm = np.convolve(s, np.ones(9) / 9, mode="same")
    vis = sm >= min_strength
    return t[vis], f[vis], s[vis]


def fit_pass(t, f, w=None, c=343.0):
    """Least-squares straight-pass fit. Returns params dict and 1-sigma uncertainties."""
    if w is None:
        w = np.ones_like(f)
    fmax, fmin = np.percentile(f, 95), np.percentile(f, 5)
    R = fmax / fmin
    v0 = np.clip(c * (R - 1) / (R + 1), 20, 120)
    f00 = 2 * fmax * fmin / (fmax + fmin)
    i = int(np.argmin(np.abs(f - f00)))
    t00 = t[i]
    slope = np.gradient(f, t)
    sl = np.median(np.abs(slope[max(0, i - 4) : i + 5])) + 1e-3
    d0 = np.clip(f00 * v0 * v0 / (c * sl), 100, 8000)
    sw = np.sqrt(w / w.max())

    def resid(p):
        return (straight_line(t, p[0], p[1], p[2], p[3], c) - f) * sw

    best = None
    for dscale in (0.5, 1.0, 2.0):
        r = least_squares(
            resid,
            [t00, f00, v0, d0 * dscale],
            bounds=([t.min() - 60, f.min() * 0.5, 5, 30], [t.max() + 60, f.max() * 1.5, 160, 20000]),
            loss="soft_l1",
            f_scale=1.0,
            x_scale=[5, 5, 10, 300],
        )
        if best is None or r.cost < best.cost:
            best = r
    p = best.x
    res = straight_line(t, *p, c) - f
    rms = float(np.sqrt(np.average(res**2, weights=w)))
    J = best.jac
    try:
        cov = np.linalg.inv(J.T @ J) * rms**2
        sig = np.sqrt(np.maximum(np.diag(cov), 0))
    except np.linalg.LinAlgError:
        sig = np.full(4, np.nan)
    return dict(t_heard=p[0], f0=p[1], speed=p[2], dist=p[3]), dict(zip(["t_heard", "f0", "speed", "dist"], sig)), rms


def measure(spec: Spectrogram, t_lo, t_hi, f_lo, f_hi, c=343.0, harmonics=(1.0,), widen=1.0, note="") -> Event:
    """Track + fit, then turn the fit into generous acceptance ranges.

    Ranges are max(3 sigma, a floor) so that real-world departures from a straight pass (turns,
    climbs, wind) do not throw out the true flight. `widen` scales every floor.
    """
    t, f, s = track_line(spec, t_lo, t_hi, f_lo, f_hi)
    if len(t) < 20:
        raise ValueError(f"tone line not found in {t_lo}-{t_hi}s / {f_lo}-{f_hi}Hz (only {len(t)} frames)")
    p, sig, rms = fit_pass(t, f, s, c)

    ev, _ = _event_from_fit(p, sig, rms, t, f, widen=widen)
    _attach_line(ev, t, f)
    ev.band = (float(f_lo), float(f_hi))
    ev.harmonics = list(harmonics)
    ev.note = note
    return ev, (t, f, s)


# ----------------------------------------------------------------------------- automatic scan

def _peaks(spec: Spectrogram, thr):
    V = spec.values
    up = V[1:-1] > V[:-2]
    dn = V[1:-1] >= V[2:]
    m = up & dn & (V[1:-1] >= thr)
    fi, ti = np.nonzero(m)
    return fi + 1, ti


def _link(spec: Spectrogram, thr=2.5, max_slope=6.0, max_gap_s=2.5, min_len_s=12.0):
    """Greedy multi-line tracker: link spectral peaks frame to frame into continuous lines."""
    fi, ti = _peaks(spec, thr)
    order = np.argsort(ti, kind="stable")
    fi, ti = fi[order], ti[order]
    frames = np.split(np.arange(len(ti)), np.flatnonzero(np.diff(ti)) + 1)
    df, dt = spec.df, spec.dt
    gap = int(max_gap_s / dt)
    active = []  # each: dict(t=[...], f=[...], s=[...], last=frame)
    done = []
    for fr in frames:
        if len(fr) == 0:
            continue
        t_idx = ti[fr[0]]
        fs = spec.f0 + fi[fr] * df
        ss = spec.values[fi[fr], t_idx]
        used = np.zeros(len(fr), bool)
        # oldest/longest tracks pick first
        active.sort(key=lambda a: -len(a["t"]))
        for a in active:
            k = t_idx - a["last"]
            if len(a["f"]) >= 4:
                slope = (a["f"][-1] - a["f"][-4]) / max(a["t"][-1] - a["t"][-4], 1e-6)
            else:
                slope = 0.0
            pred = a["f"][-1] + slope * k * dt
            tol = 1.0 + max_slope * k * dt * 0.5
            dist = np.abs(fs - pred)
            dist[used] = np.inf
            j = int(np.argmin(dist))
            if dist[j] <= tol:
                used[j] = True
                a["t"].append(spec.t0 + t_idx * dt)
                a["f"].append(fs[j])
                a["s"].append(ss[j])
                a["last"] = t_idx
        for j in np.flatnonzero(~used):
            active.append(dict(t=[spec.t0 + t_idx * dt], f=[fs[j]], s=[ss[j]], last=t_idx))
        still = []
        for a in active:
            (still if t_idx - a["last"] <= gap else done).append(a)
        active = still
    done += active
    out = []
    for a in done:
        if a["t"][-1] - a["t"][0] >= min_len_s:
            out.append((np.array(a["t"]), np.array(a["f"]), np.array(a["s"])))
    return out


def _merge_fragments(lines, max_gap_s=8.0, slope_s=8.0):
    """Join lines that continue each other: B starts at most max_gap_s after A ends, at the frequency A's
    recent slope predicts (within 2 Hz + 0.5 Hz per second of gap), falling at a similar rate. A tone line is
    often broken where it crosses another line or fades for a few seconds; the pieces are one pass."""
    lines = sorted(lines, key=lambda l: l[0][0])
    starts = np.array([l[0][0] for l in lines])
    used = np.zeros(len(lines), bool)
    out = []
    for i in range(len(lines)):
        if used[i]:
            continue
        t, f, s = lines[i]
        used[i] = True
        while True:
            w = t >= t[-1] - slope_s
            slope = np.polyfit(t[w], f[w], 1)[0] if w.sum() >= 4 and t[w][-1] > t[w][0] else 0.0
            lo = np.searchsorted(starts, t[-1] - 1.0)
            hi = np.searchsorted(starts, t[-1] + max_gap_s, side="right")
            best, best_err = -1, np.inf
            for j in range(lo, hi):
                if used[j]:
                    continue
                tj, fj, _ = lines[j]
                gap = tj[0] - t[-1]
                pred = f[-1] + slope * gap
                err = abs(fj[0] - pred)
                wj = tj <= tj[0] + slope_s
                sj = np.polyfit(tj[wj], fj[wj], 1)[0] if wj.sum() >= 4 and tj[wj][-1] > tj[wj][0] else slope
                if err <= 2.0 + 0.5 * max(gap, 0.0) and abs(sj - slope) <= 0.6 + 0.5 * abs(slope) and err < best_err:
                    best, best_err = j, err
            if best < 0:
                break
            tj, fj, sj_ = lines[best]
            keep = tj > t[-1]
            t, f, s = np.r_[t, tj[keep]], np.r_[f, fj[keep]], np.r_[s, sj_[keep]]
            used[best] = True
        out.append((t, f, s))
    return out


def _descents(t, f, min_drop=0.03, margin_s=15.0, smooth_s=4.0):
    """Split one tone line into falling stretches (approach -> recede). Returns index slices."""
    if len(t) < 8:
        return []
    dt = np.median(np.diff(t))
    k = max(1, int(smooth_s / dt))
    fs = np.convolve(np.pad(f, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    # turning points with hysteresis of min_drop/3
    hyst = min_drop / 3 * np.median(f)
    ext = [0]
    direction = 0
    for i in range(1, len(fs)):
        ref = fs[ext[-1]]
        if direction >= 0 and fs[i] > ref and direction == 1:
            ext[-1] = i
        elif direction <= 0 and fs[i] < ref and direction == -1:
            ext[-1] = i
        elif fs[i] - ref > hyst and direction != 1:
            if direction == 0:
                ext[-1] = int(np.argmin(fs[: i + 1]))
            ext.append(i)
            direction = 1
        elif ref - fs[i] > hyst and direction != -1:
            if direction == 0:
                ext[-1] = int(np.argmax(fs[: i + 1]))
            ext.append(i)
            direction = -1
    out = []
    for a, b in zip(ext[:-1], ext[1:]):
        if fs[b] < fs[a] and (fs[a] - fs[b]) / fs[a] >= min_drop:
            lo = np.searchsorted(t, t[a] - margin_s)
            hi = np.searchsorted(t, t[b] + margin_s, side="right")
            out.append(slice(lo, hi))
    return out


def scan(spec: Spectrogram, c=343.0, thr=2.5, min_len_s=12.0, max_rms_hz=2.5, log=None):
    """Find aircraft passes anywhere in the recording.

    1. link spectral peaks into continuous tone lines;
    2. cut every line into falling stretches (each one an approach-then-recede);
    3. fit each stretch with the straight-pass model. Clean fits give tight ranges; stretches
       that fall but do not fit a straight pass (turning, circling aircraft) are kept as
       "irregular" with wide ranges and left for the exact-trajectory scoring to judge;
    4. merge duplicates and group harmonics of one aircraft.
    """
    lines = _link(spec, thr=thr, min_len_s=4.0)
    lines = [l for l in _merge_fragments(lines) if l[0][-1] - l[0][0] >= min_len_s]
    pieces = []
    for t, f, s in lines:
        for sl in _descents(t, f):
            tt, ff, ss = t[sl], f[sl], s[sl]
            if tt[-1] - tt[0] < min_len_s:
                continue
            try:
                p, sig, rms = fit_pass(tt, ff, ss, c)
            except Exception:
                continue
            regular = (rms <= max_rms_hz and 15 < p["speed"] < 155 and 40 < p["dist"] < 15000
                       and tt.min() - 10 <= p["t_heard"] <= tt.max() + 10)
            if not regular:
                p, sig, rms = _irregular_guess(tt, ff), None, rms
            pieces.append(dict(t=tt, f=ff, s=ss, p=p, sig=sig, rms=rms, regular=regular))
    if log:
        log(f"{len(lines)} tone lines, {len(pieces)} falling stretches")

    # duplicates (same stretch found twice) and harmonics: same heard time, simple f0 ratio. A clean straight-pass
    # stretch only joins an irregular one if both change frequency in the same proportion (harmonics do), so a
    # long flat tone cannot swallow a real pass heard at the same time.
    pieces.sort(key=lambda d: -(len(d["t"]) * np.mean(d["s"]) * (2 if d["regular"] else 1)))
    groups = []
    for d in pieces:
        for g in groups:
            q = g[0]["p"]
            tol = 5 if (g[0]["regular"] and d["regular"]) else 12
            close = abs(q["t_heard"] - d["p"]["t_heard"]) < tol
            harmonic = close and _simple_ratio(d["p"]["f0"] / q["f0"]) and (
                g[0]["regular"] or not d["regular"] or _same_shape(g[0], d))
            if harmonic or (abs(q["t_heard"] - d["p"]["t_heard"]) < 12 and _parallel(g[0], d)):
                g.append(d)
                break
        else:
            groups.append([d])

    events = []
    for g in groups:
        prim = g[0]
        t, f, s = prim["t"], prim["f"], prim["s"]
        p = prim["p"]
        if prim["regular"]:
            ev, _ = _event_from_fit(p, prim["sig"], prim["rms"], t, f)
        else:
            ev = _irregular_event(p, t, f, prim["rms"])
        ratios = {round(float(h["p"]["f0"] / p["f0"]), 3) for h in g[1:]}
        ev.harmonics = sorted({1.0} | {r for r in ratios if abs(r - 1) > 0.02})
        span = t.max() - t.min()
        ev.quality = float(span * np.mean(s))
        before = p["t_heard"] - t.min()
        after = t.max() - p["t_heard"]
        ev.confident = bool(span >= 25 and min(before, after) >= 5
                            and (np.percentile(f, 95) - np.percentile(f, 5)) / p["f0"] >= 0.05)
        ev.note = "regular" if prim["regular"] else "irregular (turning/climbing?): wide ranges"
        _attach_line(ev, t, f)
        events.append(ev)
    events.sort(key=lambda e: e.t_heard)
    return events


def _same_shape(a, b, min_overlap_s=8.0):
    """Do two stretches change frequency in the same proportion where they overlap in time?"""
    lo = max(a["t"][0], b["t"][0])
    hi = min(a["t"][-1], b["t"][-1])
    if hi - lo < min_overlap_s:
        return False
    da = np.log(np.interp(hi, a["t"], a["f"]) / np.interp(lo, a["t"], a["f"]))
    db = np.log(np.interp(hi, b["t"], b["f"]) / np.interp(lo, b["t"], b["f"]))
    return abs(da - db) <= 0.02 + 0.15 * max(abs(da), abs(db))


def _parallel(a, b, min_overlap_s=8.0, min_fall=0.03, max_sd=0.006):
    """One aircraft heard through two tones (engine and propeller, say, at any frequency ratio): every tone
    of one source shares the same Doppler factor, so the log frequencies of the two lines keep a constant
    difference while both fall. True when they overlap long enough, fall noticeably there, and the difference
    of their log frequencies varies by at most max_sd."""
    lo = max(a["t"][0], b["t"][0])
    hi = min(a["t"][-1], b["t"][-1])
    if hi - lo < min_overlap_s:
        return False
    tt = np.linspace(lo, hi, max(8, int(hi - lo)))
    la = np.log(np.interp(tt, a["t"], a["f"]))
    lb = np.log(np.interp(tt, b["t"], b["f"]))
    if la[0] - la[-1] < min_fall or lb[0] - lb[-1] < min_fall:
        return False
    return float(np.std(la - lb)) <= max_sd


def _irregular_guess(t, f):
    """Closest approach ~ steepest fall; rest frequency ~ frequency there."""
    dt = np.median(np.diff(t))
    k = max(1, int(3 / dt))
    fs = np.convolve(np.pad(f, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")
    i = int(np.argmin(np.gradient(fs, t)))
    fmax, fmin = f.max(), f.min()
    R = fmax / fmin
    v = float(np.clip(343 * (R - 1) / (R + 1), 15, 150))
    return dict(t_heard=float(t[i]), f0=float(fs[i]), speed=v, dist=float("nan"))


def _irregular_event(p, t, f, rms, f_pad=15.0):
    return Event(
        t_heard=p["t_heard"], f0=p["f0"], speed=p["speed"], dist=float("nan"),
        t_range=(p["t_heard"] - 12, p["t_heard"] + 12),
        f0_range=(p["f0"] * 0.95, p["f0"] * 1.05),
        speed_range=(max(10.0, p["speed"] * 0.5), 170.0),
        dist_range=(60.0, 12000.0),
        window=(float(t.min()), float(t.max())),
        band=(float(f.min() - f_pad), float(f.max() + f_pad)),
        rms_hz=float(rms),
    )


def _attach_line(ev, t, f, step=0.5):
    """Keep the tracked line, thinned to ~step seconds, for fast line-based scoring."""
    keep = np.r_[0, np.flatnonzero(np.diff(np.floor(t / step)) > 0) + 1]
    ev.line_t = [round(float(x), 3) for x in t[keep]]
    ev.line_f = [round(float(x), 3) for x in f[keep]]


def _simple_ratio(r, tol=0.02):
    """True if r is close to k/m with small integers (harmonics of one engine/propeller)."""
    for m in (1, 2, 3, 4):
        k = round(r * m)
        if k >= 1 and abs(r * m - k) / (r * m) < tol:
            return True
    return False


def _event_from_fit(p, sig, rms, t, f, widen=1.0, f_pad=15.0):
    def rng(x, sg, floor_abs):
        h = max(3 * (sg if np.isfinite(sg) else 0), floor_abs) * widen
        return (float(x - h), float(x + h))

    # A straight-pass fit is only a first guess: turns, speed changes and climbs bend the curve
    # (a turning approach can read 30%+ fast). These ranges only prune the coarse search; the
    # exact curve of each real trajectory decides later.
    v_lo, v_hi = rng(p["speed"], sig["speed"], 0.35 * p["speed"] + 5)
    d = p["dist"]
    dsig = sig["dist"] if np.isfinite(sig["dist"]) else 0
    d_lo = min(d / (2.0 * widen), d - 3 * dsig)
    d_hi = max(d * 2.0 * widen, d + 3 * dsig)
    # how far the closest-approach time moves if the rest frequency is off by 2%
    tt = np.array([p["t_heard"] - 0.5, p["t_heard"] + 0.5])
    fslope = abs(np.diff(straight_line(tt, p["t_heard"], p["f0"], p["speed"], d, 343.0))[0])
    t_floor = float(np.clip(0.02 * p["f0"] / max(fslope, 1e-3), 4.0, 40.0))
    ev = Event(
        t_heard=float(p["t_heard"]),
        f0=float(p["f0"]),
        speed=float(p["speed"]),
        dist=float(d),
        t_range=rng(p["t_heard"], sig["t_heard"], t_floor),
        f0_range=rng(p["f0"], sig["f0"], 0.015 * p["f0"]),
        speed_range=(max(v_lo, 10.0), float(v_hi)),
        dist_range=(float(max(d_lo, 30.0)), float(d_hi)),
        window=(float(t.min()), float(t.max())),
        band=(float(f.min() - f_pad), float(f.max() + f_pad)),
        rms_hz=float(rms),
    )
    return ev, None
