"""Synthetic benchmark cases.

Each case is a real place and time: the recording is synthesized from the real ADS-B traffic that passed
within 6 km (Doppler-shifted tones with random rest frequencies and harmonics, in noise), and the search
area is a box of +/- radius_km around a centre 25-45 km away from the truth, for the whole local day.
"""

# name, lat, lon, start (UTC), duration (s), time zone of the search day
CASES = [
    ("long-island", 40.75000, -73.15000, "2026-09-08T15:10:00", 1800, "America/New_York"),
    ("sf-bay", 37.66000, -122.12000, "2026-09-09T19:37:20", 1200, "America/Los_Angeles"),
    ("denver", 39.93000, -105.08000, "2026-09-07T17:12:40", 900, "America/Denver"),
    ("irvine-a", 33.66189, -117.75991, "2026-09-07T22:32:08", 600, "America/Los_Angeles"),
    ("orlando-a", 28.50547, -81.29008, "2026-09-07T14:14:51", 1800, "America/New_York"),
    ("seattle-a", 47.54615, -122.31936, "2026-09-07T15:12:27", 1200, "America/Los_Angeles"),
    ("chicago-a", 41.66006, -87.99392, "2026-09-08T18:22:07", 1800, "America/Chicago"),
    ("dallas-a", 32.90660, -97.09462, "2026-09-09T22:43:09", 1200, "America/Chicago"),
    ("boston-a", 42.44427, -71.17740, "2026-09-08T22:15:25", 600, "America/New_York"),
    ("austin-a", 30.22998, -97.67173, "2026-09-09T19:20:51", 1800, "America/Chicago"),
    ("portland-a", 45.52174, -122.79939, "2026-09-07T15:19:32", 1800, "America/Los_Angeles"),
    ("irvine-b", 33.62186, -117.89030, "2026-09-08T14:49:48", 1200, "America/Los_Angeles"),
    ("orlando-b", 28.44882, -81.34501, "2026-09-08T19:55:18", 1200, "America/New_York"),
    ("seattle-b", 47.52709, -122.29544, "2026-09-07T22:42:37", 1800, "America/Los_Angeles"),
    ("chicago-b", 41.59535, -87.94779, "2026-09-09T22:01:52", 1800, "America/Chicago"),
    ("dallas-b", 32.89535, -97.01125, "2026-09-09T14:56:06", 1800, "America/Chicago"),
    ("boston-b", 42.45588, -71.17111, "2026-09-07T17:39:35", 1200, "America/New_York"),
    ("austin-b", 30.29011, -97.62155, "2026-09-08T21:27:36", 1200, "America/Chicago"),
    ("portland-b", 45.52365, -122.90676, "2026-09-08T14:54:30", 600, "America/Los_Angeles"),
    # the place read out at the end of the colsto video (a synthetic recording there, not the video's recording)
    ("oyster-bay", 40.87000, -73.50000, "2026-09-08T20:26:47", 1800, "America/New_York"),
]
RADIUS_KM = 80.0
