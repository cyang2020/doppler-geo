"""End-to-end: audio -> passes -> candidate search (kinematic, or straight-pass voting when fewer than 3 passes show a clear fastest fall) -> fine scoring with real
trajectories -> fill in the remaining passes -> refine the winner -> report."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import adsb, audio, coarse, fine, kin, measure, peaks, score, shape
from .dem import DEM
from .physics import speed_of_sound


def utc_days(t_lo, t_hi):
    d0 = int(np.floor(t_lo / 86400))
    d1 = int(np.floor(t_hi / 86400))
    return [pd.Timestamp(d * 86400, unit="s").strftime("%Y-%m-%d") for d in range(d0, d1 + 1)]


def choose_events(events, n_max=12, with_peaks=False):
    """Passes to search with: clear (confident) ones that fit a straight pass (regular), clearest first; with_peaks,
    the remaining places up to n_max go to irregular ones with a clear fastest fall (manoeuvring aircraft; their
    lines are more often noise, so they only fill up). Two lines with the same fastest fall (time within 3 s,
    rate within 10 %) are one aircraft heard twice; the clearer is kept."""
    regular = sorted([e for e in events if e.confident and e.note.startswith("regular")], key=lambda e: -e.quality)
    other = []
    if with_peaks:
        # an irregular line must still fall overall (approach, then recede) to count as a pass
        other = sorted([e for e in events if e.confident and not e.note.startswith("regular") and peaks.audio_peak(e).ok],
                       key=lambda e: -e.quality)
    pool = regular + other
    kept, seen = [], []
    for e in pool:
        ap = peaks.audio_peak(e)
        if ap.ok and any(abs(ap.t - t) < 3.0 and abs(np.log(ap.rate / r)) < 0.1 for t, r in seen):
            continue
        if ap.ok:
            seen.append((ap.t, ap.rate))
        kept.append(e)
        if len(kept) == n_max:
            break
    return sorted(kept, key=lambda e: e.t_heard)


def locate(audio_in, bbox, t0_range, out_dir=None, adsb_cache=None, events=None, n_events=12, temp_c=20.0,
           grid_m=200.0, min_votes=3, n_full=6000, classes=None, log=print):
    """Search bbox (lat_min, lat_max, lon_min, lon_max) for the place and start time (UTC epoch within
    t0_range) of a recording. audio_in: a file path, or mono samples at audio.DEFAULT_SR. With out_dir, the
    result, ranked places, a map and curve overlays are written there. Returns a JSON-ready dict."""
    out = None
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
    if adsb_cache is None:
        from .config import adsb_dir

        adsb_cache = adsb_dir()
    T = time.time()
    timing = {}
    c = speed_of_sound(temp_c)
    x = np.asarray(audio_in, np.float32) if isinstance(audio_in, np.ndarray) else audio.load(audio_in)
    duration = len(x) / audio.DEFAULT_SR
    spec = audio.spectrogram(x, audio.DEFAULT_SR)
    all_events = measure.scan(spec, c=c, log=log)
    if events is None:
        use = choose_events(all_events, n_events, with_peaks=True)
        if sum(peaks.audio_peak(e).ok for e in use) < 3:
            use = choose_events(all_events, n_events)  # straight-pass search: regular passes only
    else:
        use = [all_events[i] for i in events]
    if len(use) < 2:
        raise RuntimeError(f"only {len(use)} clear aircraft pass(es) in the audio; need at least 2")
    if out is not None:
        (out / "events.json").write_text(json.dumps({"all": [asdict(e) for e in all_events],
                                                     "used": [all_events.index(e) for e in use]}, indent=1))
    log(f"{len(use)} passes used: " + ", ".join(f"{e.t_heard:.0f}s" for e in use))
    timing["audio"] = time.time() - T

    days = utc_days(t0_range[0] - 900, t0_range[1] + duration + 900)
    day_dirs = [adsb.ensure_day(d, adsb_cache, log=log) for d in days]

    n_peak = sum(peaks.audio_peak(e).ok for e in use)
    if n_peak >= 3:
        # passes with a clear fastest fall: kinematic search (exact for manoeuvring aircraft)
        t1 = time.time()
        kcfg = kin.KinConfig(bbox=bbox, t0_range=t0_range, audio_duration=duration, c=c, classes=classes, log=log)
        cands, fl, pts, proj = kin.run(use, day_dirs, kcfg)
        timing["coarse"] = time.time() - t1
        method = "kinematic"
    else:
        t1 = time.time()
        tables = [shape.build(e, c=c) for e in use]
        timing["tables"] = time.time() - t1
        t1 = time.time()
        cfg = coarse.CoarseConfig(bbox=bbox, t0_range=t0_range, audio_duration=duration, c=c, grid_m=grid_m,
                                  min_votes=min_votes, classes=classes, log=log)
        cands, fl, pts, segs, proj = coarse.run(use, tables, day_dirs, cfg)
        timing["coarse"] = time.time() - t1
        method = "straight-pass"
    if cands is None:
        raise RuntimeError("no place in the area explains enough passes; try a larger area or time window")

    t1 = time.time()
    dem = DEM(12)
    K = len(use)
    bank = cands["bank"]
    FS = fine.FullScorer(use, bank, pts, proj, dem, c)
    if method == "kinematic" and len(cands["x"]) > n_full:
        # rank the many candidates by how well the flight matched to each pass explains it (exact curves, one
        # flight per pass), then give the best n_full to the full scoring that tries every nearby flight
        ms, _ = FS.matched(cands["x"], cands["y"], cands["T0"], cands["fid"])
        top_m = np.argsort(-ms, kind="stable")[:n_full]
        cands = {k: (v[top_m] if isinstance(v, np.ndarray) and len(v) == len(ms) else v) for k, v in cands.items()}
        timing["matched"] = time.time() - t1
    # stage A: the best coarse candidates, every pass against every nearby flight, +/- 8 s
    nA = min(n_full, len(cands["x"]))
    totA, T0A, _, _ = FS(cands["x"][:nA], cands["y"][:nA], cands["T0"][:nA], shift_s=8.0, step=2.0)
    # stage B: best 100 places at 9 points 100 m apart, +/- 3 s at 1 s
    top = _distinct(np.argsort(-totA), cands["x"][:nA], cands["y"][:nA], None, 100, min_sep=150.0)
    offs = np.array([(dx, dy) for dx in (-100, 0, 100) for dy in (-100, 0, 100)], float)
    bx = (cands["x"][top][:, None] + offs[None, :, 0]).ravel()
    by = (cands["y"][top][:, None] + offs[None, :, 1]).ravel()
    bT0 = np.repeat(T0A[top], len(offs))
    totB, T0B, partsB, fidB = FS(bx, by, bT0, shift_s=3.0, step=1.0)
    timing["fine"] = time.time() - t1

    # distinct places >= 2 km apart: the winner and its strongest rival
    t1 = time.time()
    order = _distinct(np.argsort(-totB), bx, by, None, 10, min_sep=2000.0)
    places = [dict(x=bx[i], y=by[i], T0=T0B[i], score=float(totB[i]), fid=fidB[i]) for i in order]
    timing["fill"] = time.time() - t1

    # stage D: 25 m grid around the winner
    t1 = time.time()
    w = places[0]
    g = np.arange(-300, 300.1, 25.0)
    gx, gy = np.meshgrid(w["x"] + g, w["y"] + g)
    gx, gy = gx.ravel(), gy.ravel()
    gt, gT0, gparts, gfid = FS(gx, gy, np.full(len(gx), w["T0"]), shift_s=2.0, step=1.0)
    glat, glon = proj.latlon(gx, gy)
    j = int(gt.argmax())
    w["fid"] = gfid[j]
    heat = pd.DataFrame(dict(lat=glat, lon=glon, score=gt))
    # spread: places within 0.25 points (a quarter of one pass) of the best
    close = gt >= gt[j] - 0.25
    spread = float(np.hypot(gx[close] - gx[j], gy[close] - gy[j]).max()) if close.any() else 0.0
    timing["refine"] = time.time() - t1
    w["bank"] = bank

    meta = fl.set_index("fid")
    flights = []
    for k in range(K):
        f = int(w["fid"][k])
        m = meta.loc[f] if f in meta.index else None
        flights.append(dict(pass_t=round(use[k].t_heard, 1), fid=f, score=round(float(gparts[j, k]), 3),
                            type="" if m is None else m.type, reg="" if m is None else m.reg,
                            icao="" if m is None else m.icao))
    runner = places[1]["score"] if len(places) > 1 else 0.0
    rlat, rlon = proj.latlon(places[1]["x"], places[1]["y"]) if len(places) > 1 else (None, None)
    result = dict(
        lat=float(glat[j]), lon=float(glon[j]), score=float(gt[j]), passes_used=K,
        passes_explained=int((gparts[j] >= 0.5).sum()),
        recording_start_utc=pd.Timestamp(gT0[j], unit="s").isoformat() + "Z",
        spread_m=spread,
        runner_up=dict(score=runner, lat=None if rlat is None else float(rlat), lon=None if rlon is None else float(rlon)),
        margin=float(gt[j] - runner),
        confidence=_confidence(float(gt[j]), float(gt[j] - runner), int((gparts[j] >= 0.5).sum()), K),
        flights=flights,
        method=method,
        audio_duration_s=duration,
        ranked=[dict(lat=float(proj.latlon(p["x"], p["y"])[0]), lon=float(proj.latlon(p["x"], p["y"])[1]),
                     score=p["score"], start_utc=pd.Timestamp(p["T0"], unit="s").isoformat() + "Z") for p in places],
        tracks=_tracks(pts, [f["fid"] for f in flights], gT0[j] - 300, gT0[j] + duration + 300),
    )
    result["other_passes"] = _others(result, all_events, use, _near_bank(pts, result["lat"], result["lon"], gT0[j], duration),
                                     meta, dem, c)
    timing["total"] = time.time() - T
    result["timing_s"] = {k: round(v, 1) for k, v in timing.items()}
    if out is not None:
        (out / "result.json").write_text(json.dumps(result, indent=2, default=_json))
        ranked = pd.DataFrame(result["ranked"])
        ranked.to_csv(out / "ranked.csv", index=False)
        heat.to_csv(out / "refine.csv", index=False)
        from . import report

        top = dict(result, flights=[dict(f) for f in flights])
        report.write(out, top, ranked, [dict(heat=heat)], use, spec, w["bank"], dem, c)
    log(f"best: {result['lat']:.5f}, {result['lon']:.5f} (+/- {spread:.0f} m)  score {result['score']:.2f} of {K}  "
        f"start {result['recording_start_utc']}  runner-up {runner:.2f}  confidence {result['confidence']}  "
        f"time {timing['total']:.1f}s")
    return result


def _confidence(score, margin, explained, K):
    """high: several passes explained and no rival place comes close; low: a rival explains about as much."""
    if explained >= 4 and margin >= 1.5 and score >= 0.6 * K:
        return "high"
    if explained >= 3 and margin >= 0.75 and score >= 0.4 * K:
        return "medium"
    return "low"


def _json(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return str(x)


def _distinct(order, x, y, fids, n, min_sep=0.0):
    """First n indices of `order` that are not repeats (same cell and flights, or closer than min_sep)."""
    out = []
    seen = set()
    for i in order:
        key = (round(float(x[i])), round(float(y[i])), tuple(fids[i]) if fids is not None else ())
        if key in seen:
            continue
        if min_sep > 0 and any(np.hypot(x[i] - x[j], y[i] - y[j]) < min_sep for j in out):
            continue
        seen.add(key)
        out.append(int(i))
        if len(out) == n:
            break
    return np.array(out, np.int64)


def _tracks(pts, fids, t_lo, t_hi):
    """Positions of the given flights between t_lo and t_hi, for plotting: {fid: {t, lat, lon, alt_m}}."""
    out = {}
    sel = pts[pts.fid.isin([f for f in fids if f >= 0]) & (pts.t >= t_lo) & (pts.t <= t_hi)]
    for f, g in sel.groupby("fid"):
        g = g.sort_values("t")
        alt = g["alt_geom"].where(np.isfinite(g["alt_geom"]), g["alt_baro"])
        out[str(int(f))] = dict(t=g.t.round(1).tolist(), lat=g.lat.round(6).tolist(), lon=g.lon.round(6).tolist(),
                                alt_m=[None if not np.isfinite(a) else round(float(a), 1) for a in alt])
    return out


def _near_bank(pts, lat, lon, T0, duration, radius=15000.0):
    d = radius / 111000
    dl = d / np.cos(np.radians(lat))
    m = ((pts.lat.to_numpy() > lat - d) & (pts.lat.to_numpy() < lat + d) & (pts.lon.to_numpy() > lon - dl)
         & (pts.lon.to_numpy() < lon + dl) & (pts.t.to_numpy() > T0 - 300) & (pts.t.to_numpy() < T0 + duration + 300))
    p = pts[m]
    return score.TrajectoryBank(p, p.fid.unique())


def _others(result, all_events, use, bank, meta, dem, c):
    """Passes not used by the search (unclear or irregular): which flight, if any, explains them here."""
    others = [e for e in all_events if e not in use and len(e.line_t) > 10]
    if not others or len(bank.fids) == 0:
        return []
    T0 = pd.Timestamp(result["recording_start_utc"].rstrip("Z")).timestamp()
    f, s, _ = fine.best_flights(result["lat"], result["lon"], T0, others, bank, dem, c, shift_s=3.0)
    out = []
    for e, fi, si in zip(others, f, s):
        m = meta.loc[fi] if fi in meta.index else None
        out.append(dict(t_heard=round(e.t_heard, 1), note=e.note, line_score=round(float(si), 3),
                        flight=None if m is None else dict(fid=int(fi), type=m.type, reg=m.reg, icao=m.icao)))
    return out
