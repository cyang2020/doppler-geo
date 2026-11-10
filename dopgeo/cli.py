"""Command line: `dopgeo <command> ...` (or `python -m dopgeo`)."""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np


def _area(a):
    from .api import Area

    if a.bbox:
        return Area(*a.bbox)
    if a.center:
        return Area.circle(a.center[0], a.center[1], a.radius_km)
    if a.conus:
        return Area.conus()
    sys.exit("give --center LAT LON (with --radius-km), --bbox, or --conus")


def main(argv=None):
    p = argparse.ArgumentParser(prog="dopgeo", description="Locate a recording from the Doppler tones of aircraft "
                                "flying over it, using public ADS-B history.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("locate", help="find where and when a recording was made")
    s.add_argument("audio", help="audio or video file (anything ffmpeg reads)")
    t = s.add_mutually_exclusive_group(required=True)
    t.add_argument("--date", help="local calendar day of the recording, YYYY-MM-DD (start time unknown)")
    t.add_argument("--start", help="approximate start, ISO time (UTC unless an offset is given)")
    s.add_argument("--tz", default="UTC", help="time zone of --date, e.g. America/New_York (default UTC)")
    s.add_argument("--start-tol", type=float, default=600, help="seconds of uncertainty on --start (default 600)")
    r = s.add_mutually_exclusive_group()
    r.add_argument("--center", type=float, nargs=2, metavar=("LAT", "LON"), help="centre of the search area")
    r.add_argument("--bbox", type=float, nargs=4, metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    r.add_argument("--conus", action="store_true", help="search the contiguous US (slow: hours)")
    s.add_argument("--radius-km", type=float, default=50, help="radius around --center (default 50)")
    s.add_argument("--no-jets", action="store_true", help="ignore known jet types (passes are propeller aircraft)")
    s.add_argument("--passes", type=int, nargs="*", help="indices from `dopgeo passes` to use (default: automatic)")
    s.add_argument("--max-passes", type=int, default=12, help="use at most this many passes (default 12)")
    s.add_argument("--temp", type=float, default=20.0, help="air temperature in C, for the speed of sound")
    s.add_argument("--out", help="write result.json, ranked.csv, map.html and curves.png here")
    s.add_argument("--cache", help="ADS-B cache directory (default $DOPGEO_CACHE/adsb or ~/.cache/dopgeo/adsb)")
    s.add_argument("-q", "--quiet", action="store_true", help="print only the JSON result")

    s = sub.add_parser("passes", help="list the aircraft passes found in a recording")
    s.add_argument("audio")
    s.add_argument("--json", action="store_true", help="print JSON instead of a table")
    s.add_argument("--plot", help="write an annotated spectrogram PNG here")

    s = sub.add_parser("fetch", help="download and extract the ADS-B history of UTC days")
    s.add_argument("days", nargs="+", help="YYYY-MM-DD")
    s.add_argument("--cache")

    s = sub.add_parser("synth", help="make a synthetic test recording from real ADS-B traffic")
    s.add_argument("--lat", type=float, required=True)
    s.add_argument("--lon", type=float, required=True)
    s.add_argument("--start", required=True, help="UTC ISO time of audio time 0")
    s.add_argument("--duration", type=float, default=1800)
    s.add_argument("--out", required=True, help="output .wav (a .truth.json is written next to it)")
    s.add_argument("--radius", type=float, default=6000, help="include flights passing within this many metres")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--cache")

    s = sub.add_parser("serve", help="run the HTTP API (needs the [server] extra)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    a = p.parse_args(argv)
    if a.cmd == "locate":
        from . import api

        res = api.locate(a.audio, date=a.date, tz=a.tz, start=a.start, start_tolerance_s=a.start_tol, area=_area(a),
                         exclude_jets=a.no_jets, temperature_c=a.temp, max_passes=a.max_passes, passes=a.passes,
                         adsb_cache=a.cache, output_dir=a.out, verbose=not a.quiet)
        d = res.to_dict()
        d.pop("tracks")
        d.pop("ranked")
        print(json.dumps(d, indent=2, default=float))
    elif a.cmd == "passes":
        from . import api

        ps = api.detect_passes(a.audio)
        if a.json:
            print(json.dumps([dict(vars(x), line_t=None, line_f=None) for x in ps], indent=1))
        else:
            print(" idx  time_s  rest_Hz  clear  straight  fastest_fall_s  rate(1/s)")
            for x in ps:
                ff = f"{x.fastest_fall_s:8.1f}  {x.fastest_fall_rate:.4f}" if x.fastest_fall_s is not None else "       -        -"
                print(f"{x.index:4d} {x.time_s:7.1f} {x.rest_frequency_hz:8.1f}  {str(x.clear):5}  {str(x.straight):8}  {ff}")
        if a.plot:
            from . import audio, measure

            spec = audio.spectrogram(audio.load(a.audio), audio.DEFAULT_SR)
            _plot_passes(spec, measure.scan(spec, log=None), a.plot)
    elif a.cmd == "fetch":
        from . import api

        for d in a.days:
            print(api.fetch_adsb(d, a.cache))
    elif a.cmd == "synth":
        from . import api

        truth = api.synthesize(a.lat, a.lon, a.start, a.duration, a.out, seed=a.seed, radius_m=a.radius,
                               adsb_cache=a.cache)
        print(f"{a.out}: {len(truth['passes'])} audible passes")
    elif a.cmd == "serve":
        try:
            import uvicorn
        except ImportError:
            sys.exit("the HTTP API needs the server extra: pip install 'doppler-geo[server]'")
        from .server import app

        uvicorn.run(app, host=a.host, port=a.port)


def _plot_passes(spec, evs, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    span = spec.times[-1] - spec.times[0]
    rows = int(np.ceil(span / 600))
    fig, axes = plt.subplots(rows, 1, figsize=(20, 3.6 * rows), squeeze=False)
    for r, ax in enumerate(axes[:, 0]):
        lo, hi = spec.t0 + r * 600, spec.t0 + (r + 1) * 600
        sub = spec.crop(lo, hi, 30, 450)
        ax.pcolormesh(sub.times, sub.freqs, sub.values, vmin=0, vmax=6, cmap="magma", shading="auto")
        for i, e in enumerate(evs):
            if lo <= e.t_heard <= hi:
                ax.plot(e.line_t, e.line_f, color="cyan" if e.confident else "gray", lw=1)
                ax.text(e.t_heard, e.f0 + 12, str(i), color="cyan" if e.confident else "gray", fontsize=10)
        ax.set_xlim(lo, hi)
        ax.set_ylabel("Hz")
    axes[-1, 0].set_xlabel("audio time (s)")
    plt.tight_layout()
    plt.savefig(path, dpi=55)
