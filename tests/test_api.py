import pandas as pd
import pytest

import dopgeo


def test_area_circle():
    a = dopgeo.Area.circle(40.0, -73.0, 50.0)
    assert a.lat_min < 40.0 < a.lat_max and a.lon_min < -73.0 < a.lon_max
    assert abs(a.area_km2() - 100.0 * 100.0) / 1e4 < 0.02


def test_time_window_day_respects_time_zone():
    w = dopgeo.TimeWindow.day("2026-09-08", "America/New_York", duration_s=3600)
    assert pd.Timestamp(w.start, unit="s") == pd.Timestamp("2026-09-08 04:00")  # EDT is UTC-4
    assert w.end - w.start == 86400 - 3600


def test_time_window_around():
    w = dopgeo.TimeWindow.around("2026-09-08T20:30:00Z", 600)
    assert w.end - w.start == 1200
    assert pd.Timestamp(w.start, unit="s") == pd.Timestamp("2026-09-08 20:20")


def test_locate_needs_time_and_place(one_pass):
    with pytest.raises(ValueError):
        dopgeo.locate(one_pass, center=(40.0, -73.0))
    with pytest.raises(ValueError):
        dopgeo.locate(one_pass, date="2026-09-08")
