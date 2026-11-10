"""Real recordings from YouTube, located blind.

The Ambient Archive (https://www.youtube.com/@TheAmbientArchive) publishes field recordings of parks in
Los Angeles and gives, in each description, the exact coordinates and the local start time. That makes
them a fair test: the search below is told only the local date and a large area around Los Angeles, never
the place or the time of day, and the answer is compared with the description.

    pip install yt-dlp
    python examples/youtube_cases.py            # all cases
    python examples/youtube_cases.py JqRT1PCNCqg

The audio is downloaded from YouTube (lowest-bitrate audio track; only frequencies below 1 kHz are used)
and is not redistributed with this project.
"""
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

import dopgeo
from dopgeo.plot import plot_result

# video id, title, coordinates and local start time from the description
CASES = [
    ("JqRT1PCNCqg", "Ambient Bird and Park Sounds in Pasadena - Fall 2024", 34.137111, -118.167878, "2024-11-25 09:42"),
    ("3rRfrnaruFo", "Ambient Park Sounds in Mount Washington, Los Angeles - Fall 2024", 34.100614, -118.222278,
     "2024-12-02 16:17"),
    ("5PAHqKy-vRo", "Summer Sounds of a Los Angeles Park", 34.053908, -118.216922, "2025-08-26 18:33"),
    ("UVBbkumxkek", "Ambient Nature Park Sounds in Pasadena - Fall 2024", 34.137111, -118.167878, "2024-11-25 09:05"),
    ("8YW6H7uB_sg", "Ambient Park Sounds in Elysian Park, Los Angeles - Fall 2024", 34.082142, -118.244281,
     "2024-11-24 17:25"),
]
TZ = "America/Los_Angeles"
CENTER, RADIUS_KM = (34.30, -117.95), 80.0  # a 160 km box over the Los Angeles basin, centred ~30 km off
OUT = Path("youtube_cases")


def fetch(video_id):
    OUT.mkdir(exist_ok=True)
    have = list(OUT.glob(f"{video_id}.*"))
    if have:
        return have[0]
    subprocess.run(["yt-dlp", "-f", "249/250/139/bestaudio", "-o", str(OUT / f"{video_id}.%(ext)s"),
                    f"https://www.youtube.com/watch?v={video_id}"], check=True)
    return next(OUT.glob(f"{video_id}.*"))


def main(ids):
    for vid, title, lat, lon, local_start in CASES:
        if ids and vid not in ids:
            continue
        start = pd.Timestamp(local_start).tz_localize(TZ)
        r = dopgeo.locate(fetch(vid), date=start.strftime("%Y-%m-%d"), tz=TZ, center=CENTER, radius_km=RADIUS_KM,
                          output_dir=OUT / vid)
        err = math.hypot((r.lat - lat) * 111000, (r.lon - lon) * 111000 * math.cos(math.radians(lat)))
        dt = (pd.Timestamp(r.recording_start) - start.tz_convert("UTC")).total_seconds()
        print(f"{vid}  {title}\n    found {r.lat:.5f}, {r.lon:.5f}: {err:.0f} m from the stated place; start "
              f"{r.recording_start:%Y-%m-%d %H:%M:%S} UTC ({dt:+.0f} s from the stated minute); confidence "
              f"{r.confidence}; {r.passes_explained} of {r.passes_used} passes explained; {r.timing_s['total']:.0f} s")
        plot_result(r, OUT / f"{vid}.png", truth=(lat, lon), title=title)


if __name__ == "__main__":
    main(sys.argv[1:])
