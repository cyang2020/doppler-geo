"""Where downloaded data is kept.

Everything fetched from the internet (ADS-B history, terrain tiles, the aircraft type table) is cached
under one directory: $DOPGEO_CACHE if set, else ~/.cache/dopgeo.
"""
from __future__ import annotations

import os
from pathlib import Path


def cache_dir() -> Path:
    return Path(os.environ.get("DOPGEO_CACHE", Path.home() / ".cache" / "dopgeo"))


def adsb_dir() -> Path:
    """Extracted ADS-B history, one sub-directory per UTC day (about 1 GB per day for the whole world)."""
    return cache_dir() / "adsb"
