"""Synthetic recordings for end-to-end testing: what a microphone at (lat, lon) would have picked
up from every nearby aircraft in the ADS-B data, as Doppler-shifted tones in noise."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from . import adsb
from .dem import DEM
from .physics import Trajectory
from .score import observers


def synthesize(day_dirs, lat, lon, t0, duration, out_wav, sr=2000, radius=6000.0, classes=None, seed=0,
               noise=1.0, ref_amp=0.6, ref_dist=1000.0, c=343.0, log=print):
    rng = np.random.default_rng(seed)
    dlat = radius / 111000 + 0.02
    dlon = radius / (111000 * np.cos(np.radians(lat))) + 0.02
    fl, pts = adsb.load(day_dirs, bbox=(lat - dlat, lat + dlat, lon - dlon, lon + dlon), t_range=(t0 - 300, t0 + duration + 300),
                        classes=classes)
    o = observers([lat], [lon], DEM(12))[0]
    n = int(duration * sr)
    t_obs = t0 + np.arange(n) / sr
    y = rng.normal(0, 0.05 * noise, n)
    coarse = t0 + np.arange(0, duration + 0.05, 0.05)
    passes = []
    pts = pts.assign(h=adsb.fill_geometric_altitude(pts))
    for fid, g in pts.groupby("fid"):
        if len(g) < 4:
            continue
        tr = Trajectory(g.t, g.lat, g.lon, g.h, g.gs, g.track, g.vrate)
        gg, r, valid = tr.doppler_factor(o, coarse, c)
        if not valid.any() or r[valid].min() > radius:
            continue
        f0 = rng.uniform(70, 200)
        harm = [1, 2] if rng.random() < 0.6 else [1]
        amp = ref_amp * ref_dist / np.maximum(r, 50) * valid
        G = np.interp(t_obs, coarse, gg)
        A = np.interp(t_obs, coarse, amp)
        for h in harm:
            ph = 2 * np.pi * np.cumsum(f0 * h * G) / sr
            y += A / h * np.sin(ph + rng.uniform(0, 2 * np.pi))
        k = int(np.argmin(np.where(valid, r, np.inf)))
        row = fl[fl.fid == fid].iloc[0]
        passes.append(dict(fid=int(fid), type=row.type, reg=row.reg, f0=f0, harmonics=harm, min_dist=float(r[k]),
                           t_closest_audio=float(coarse[k] - t0)))
    y = y / max(1.0, np.abs(y).max() * 1.05)
    sf.write(out_wav, y.astype(np.float32), sr)
    truth = dict(lat=lat, lon=lon, t0=t0, duration=duration, passes=sorted(passes, key=lambda p: p["t_closest_audio"]))
    Path(out_wav).with_suffix(".truth.json").write_text(json.dumps(truth, indent=2))
    log(f"{len(passes)} audible passes synthesized")
    return truth
