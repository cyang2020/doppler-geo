"""End to end on a synthetic recording. Needs the ADS-B history of 2026-09-09 in the cache (about 1 GB,
downloaded with `dopgeo fetch 2026-09-09`), so it only runs with DOPGEO_INTEGRATION=1."""
import math
import os

import pytest

import dopgeo

pytestmark = pytest.mark.integration


@pytest.mark.skipif(os.environ.get("DOPGEO_INTEGRATION") != "1", reason="set DOPGEO_INTEGRATION=1 to run")
def test_synthetic_recording_is_located(tmp_path):
    lat, lon, start = 30.29011, -97.62155, "2026-09-08T21:27:36"
    wav = tmp_path / "rec.wav"
    truth = dopgeo.synthesize(lat, lon, start, 1200, wav, seed=7)
    r = dopgeo.locate(wav, date="2026-09-08", tz="America/Chicago", center=(30.5, -97.4), radius_km=60)
    err = math.hypot((r.lat - lat) * 111000, (r.lon - lon) * 111000 * math.cos(math.radians(lat)))
    assert err < 500
    assert abs(r.recording_start.timestamp() - truth["t0"]) < 5
