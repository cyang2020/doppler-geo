"""Run the synthetic benchmark: synthesize each case (once), locate it, compare with the truth.

    python benchmarks/run.py                 # all cases
    python benchmarks/run.py dallas-b sf-bay # some cases

Recordings and per-case outputs go to benchmarks/out/ (not versioned); the summary is written to
benchmarks/results.json and printed as a Markdown table.
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import dopgeo
from cases import CASES, RADIUS_KM

OUT = Path(__file__).parent / "out"


def offset_center(lat, lon, i):
    rng = np.random.default_rng(1000 + i)
    ang, dist = rng.uniform(0, 2 * np.pi), rng.uniform(25, 45)
    return lat + dist * np.sin(ang) / 111.0, lon + dist * np.cos(ang) / (111.0 * np.cos(np.radians(lat)))


def main(names):
    OUT.mkdir(exist_ok=True)
    rows = []
    res_path = Path(__file__).parent / "results.json"
    old = {r["name"]: r for r in json.loads(res_path.read_text())} if res_path.exists() else {}
    for i, (name, lat, lon, start, dur, tz) in enumerate(CASES):
        if names and name not in names:
            continue
        wav = OUT / f"{name}.wav"
        if not wav.exists():
            dopgeo.synthesize(lat, lon, start, dur, wav, seed=100 + i)
        truth = json.loads(wav.with_suffix(".truth.json").read_text())
        clat, clon = offset_center(lat, lon, i)
        local_day = pd.Timestamp(start).tz_localize("UTC").tz_convert(tz).strftime("%Y-%m-%d")
        t = time.time()
        row = dict(name=name, lat=lat, lon=lon, passes_in_audio=len(truth["passes"]), duration_s=dur,
                   center=[round(clat, 4), round(clon, 4)], day=local_day)
        try:
            r = dopgeo.locate(wav, date=local_day, tz=tz, center=(clat, clon), radius_km=RADIUS_KM,
                              output_dir=OUT / name)
            err = math.hypot((r.lat - lat) * 111000, (r.lon - lon) * 111000 * math.cos(math.radians(lat)))
            row.update(found=[round(r.lat, 5), round(r.lon, 5)], error_m=round(err), start_error_s=round(
                r.recording_start.timestamp() - truth["t0"], 1), confidence=r.confidence, passes_used=r.passes_used,
                passes_explained=r.passes_explained, method=r.method, seconds=round(r.timing_s["total"], 1))
        except RuntimeError as e:
            row.update(refused=str(e), seconds=round(time.time() - t, 1))
        rows.append(row)
        old[name] = row
        print(json.dumps(row), flush=True)
    res_path.write_text(json.dumps([old[c[0]] for c in CASES if c[0] in old], indent=1))
    print()
    print("| case | passes in audio | error | start error | confidence | passes used / explained | time |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        if "refused" in r:
            print(f"| {r['name']} | {r['passes_in_audio']} | refused: {r['refused']} | | | | {r['seconds']} s |")
        else:
            print(f"| {r['name']} | {r['passes_in_audio']} | {r['error_m']} m | {r['start_error_s']:+.1f} s | "
                  f"{r['confidence']} | {r['passes_used']} / {r['passes_explained']} | {r['seconds']} s |")


if __name__ == "__main__":
    main(sys.argv[1:])
