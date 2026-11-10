"""Locate a recording and draw the result.

    python examples/locate_recording.py recording.wav 2026-09-08 America/New_York 40.8 -73.4 80

The ADS-B history of the needed days is downloaded on first use (about 3.5 GB per UTC day, kept as
about 1 GB in ~/.cache/dopgeo/adsb).
"""
import sys

import dopgeo
from dopgeo.plot import plot_result

audio, date, tz, lat, lon, radius = sys.argv[1], sys.argv[2], sys.argv[3], *map(float, sys.argv[4:7])

for p in dopgeo.detect_passes(audio):
    if p.clear:
        ff = f"fastest fall at {p.fastest_fall_s:.0f} s" if p.fastest_fall_s is not None else "no clear fastest fall"
        print(f"pass at {p.time_s:7.1f} s, {p.rest_frequency_hz:5.1f} Hz, {'straight' if p.straight else 'irregular'}, {ff}")

result = dopgeo.locate(audio, date=date, tz=tz, center=(lat, lon), radius_km=radius, output_dir="out", verbose=True)
print(f"\n{result.lat:.5f}, {result.lon:.5f}  start {result.recording_start:%Y-%m-%d %H:%M:%S} UTC  "
      f"confidence {result.confidence}  ({result.passes_explained} of {result.passes_used} passes explained)")
for m in result.matches:
    print(f"  pass at {m.audio_time_s:7.1f} s: {m.aircraft_type or '?':5} {m.registration:8} fit {m.fit:.2f}")
print(result.map_url())
plot_result(result, "out/result.png")
