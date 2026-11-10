"""Coordinate helpers: WGS84 ECEF, CONUS Albers equal-area grid, geoid heights."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pyproj

FT = 0.3048
KT = 0.514444

ALBERS = "EPSG:5070"  # NAD83 / Conus Albers, metres

_A = 6378137.0
_F = 1 / 298.257223563
_E2 = _F * (2 - _F)


def ecef(lat, lon, h):
    """lat/lon in degrees, h = height above the WGS84 ellipsoid in metres."""
    lat = np.radians(lat)
    lon = np.radians(lon)
    n = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
    x = (n + h) * np.cos(lat) * np.cos(lon)
    y = (n + h) * np.cos(lat) * np.sin(lon)
    z = (n * (1 - _E2) + h) * np.sin(lat)
    return np.stack([x, y, z], axis=-1)


def enu_to_ecef_vec(lat, lon, e, n, u):
    """Rotate local east/north/up vectors into ECEF."""
    lat = np.radians(lat)
    lon = np.radians(lon)
    sl, cl = np.sin(lat), np.cos(lat)
    so, co = np.sin(lon), np.cos(lon)
    x = -so * e - sl * co * n + cl * co * u
    y = co * e - sl * so * n + cl * so * u
    z = cl * n + sl * u
    return np.stack([x, y, z], axis=-1)


@lru_cache(maxsize=1)
def _to_albers():
    return pyproj.Transformer.from_crs("EPSG:4326", ALBERS, always_xy=True)


@lru_cache(maxsize=1)
def _from_albers():
    return pyproj.Transformer.from_crs(ALBERS, "EPSG:4326", always_xy=True)


def to_albers(lat, lon):
    x, y = _to_albers().transform(np.asarray(lon), np.asarray(lat))
    return np.asarray(x), np.asarray(y)


def from_albers(x, y):
    lon, lat = _from_albers().transform(np.asarray(x), np.asarray(y))
    return np.asarray(lat), np.asarray(lon)


@lru_cache(maxsize=1)
def _geoid():
    pyproj.network.set_network_enabled(True)
    return pyproj.Transformer.from_crs("EPSG:4979", "EPSG:4326+5773", always_xy=True)


def geoid_undulation(lat, lon):
    """EGM96 geoid height N (metres): ellipsoid height = orthometric height + N.

    Falls back to -30 m (a typical CONUS value) when the PROJ grid is unreachable.
    """
    lat = np.atleast_1d(np.asarray(lat, float))
    lon = np.atleast_1d(np.asarray(lon, float))
    try:
        _, _, h = _geoid().transform(lon, lat, np.zeros_like(lat))
        h = np.asarray(h)
        if np.all(np.isfinite(h)):
            return -h
    except Exception:
        pass
    return np.full_like(lat, -30.0)


def haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(a))
