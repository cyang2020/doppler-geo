"""Figures for the README.

    python examples/youtube_cases.py    # downloads the recordings, locates them, draws the maps
    python docs/make_figures.py         # copies the maps and draws the spectrogram comparison
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dopgeo import adsb, audio, pipeline, score
from dopgeo.config import adsb_dir
from dopgeo.dem import DEM
from dopgeo.measure import Event

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "out"
IMG = ROOT / "docs" / "images"


def curves(name, n=4, path=None, run_dir=None, audio_file=None):
    """Spectrogram crops of the best-explained passes, with the tracked line and the exact Doppler curve
    of the flight that explains it at the found place and time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir) if run_dir else OUT / name
    res = json.loads((run_dir / "result.json").read_text())
    ev = json.loads((run_dir / "events.json").read_text())
    used = [ev["all"][i] for i in ev["used"]]
    x = audio.load(audio_file or OUT / f"{name}.wav")
    spec = audio.spectrogram(x, audio.DEFAULT_SR)
    T0 = pd.Timestamp(res["recording_start_utc"].rstrip("Z")).timestamp()
    days = [adsb_dir() / d for d in pipeline.utc_days(T0 - 900, T0 + res["audio_duration_s"] + 900)]
    fids = [f["fid"] for f in res["flights"] if f["fid"] >= 0]
    fl, pts = adsb.load(days, bbox=(res["lat"] - .3, res["lat"] + .3, res["lon"] - .4, res["lon"] + .4),
                        t_range=(T0 - 900, T0 + res["audio_duration_s"] + 900))
    bank = score.TrajectoryBank(pts[pts.fid.isin(fids)], fids)
    obs = score.observers([res["lat"]], [res["lon"]], DEM(12))[0]
    best = sorted(range(len(used)), key=lambda k: -res["flights"][k]["score"])[:n]
    best.sort(key=lambda k: used[k]["t_heard"])
    fig, axes = plt.subplots(1, len(best), figsize=(4.2 * len(best), 3.8), squeeze=False)
    for ax, k in zip(axes[0], best):
        e = Event(**{kk: (tuple(v) if isinstance(v, list) and kk in ("t_range", "f0_range", "speed_range",
                                                                     "dist_range", "window", "band") else v)
                     for kk, v in used[k].items()})
        f = res["flights"][k]
        lo, hi = e.window[0] - 20, e.window[1] + 20
        flo, fhi = min(e.line_f) * 0.8, max(e.line_f) * 1.2
        sub = spec.crop(lo, hi, flo, fhi)
        ax.pcolormesh(sub.times, sub.freqs, sub.values, vmin=0, vmax=6, cmap="magma", shading="auto")
        ax.plot(e.line_t, e.line_f, color="cyan", lw=1.0, label="tracked line")
        i = int(np.searchsorted(bank.fids, f["fid"]))
        if i < len(bank.fids) and bank.fids[i] == f["fid"]:
            t = np.arange(lo, hi, 0.5)
            g = np.empty(len(t))
            ok = np.zeros(len(t), bool)
            score._curve(bank.t, bank.pos, bank.vel, bank.start[i], bank.start[i + 1], obs, T0 + t[0], 0.5, len(t),
                         343.2, g, ok)
            lt, lf = np.asarray(e.line_t), np.asarray(e.line_f)
            gi = np.interp(lt, t, g)
            f0 = float((lf * gi).sum() / (gi * gi).sum())
            ax.plot(t[ok], f0 * g[ok], color="white", lw=1.2, ls="--", label="exact curve of the flight")
        ax.set_title(f"pass at {e.t_heard:.0f} s: {f['type'] or '?'} {f['reg']}  (fit {f['score']:.2f})", fontsize=9)
        ax.set_xlabel("audio time (s)")
        ax.set_ylabel("Hz")
    axes[0, 0].legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path or IMG / f"curves_{name}.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    # run examples/youtube_cases.py first (it writes youtube_cases/<id>/ and the maps)
    IMG.mkdir(parents=True, exist_ok=True)
    yt = Path("youtube_cases")
    for vid in ("JqRT1PCNCqg", "3rRfrnaruFo", "5PAHqKy-vRo"):
        if (yt / f"{vid}.png").exists():
            (IMG / f"youtube_{vid}.png").write_bytes((yt / f"{vid}.png").read_bytes())
    audio_file = next(yt.glob("JqRT1PCNCqg.*[!g]"), None)
    if audio_file is not None:
        curves("pasadena", n=4, path=IMG / "youtube_JqRT1PCNCqg_curves.png", run_dir=yt / "JqRT1PCNCqg",
               audio_file=audio_file)
    print("figures written to", IMG)
