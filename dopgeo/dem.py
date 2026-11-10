"""Ground elevation from the public AWS Terrain Tiles (Terrarium PNG encoding, global).

elevation_m = R * 256 + G + B / 256 - 32768 (orthometric, metres). Water is clamped to 0 because
the dataset carries bathymetry and a listener on water sits at the surface.
Tiles are cached on disk under ~/.cache/dopgeo/terrarium.
"""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from PIL import Image

URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
CACHE = Path(os.environ.get("DOPGEO_CACHE", Path.home() / ".cache" / "dopgeo")) / "terrarium"


class DEM:
    def __init__(self, zoom: int = 12, cache: Path = CACHE, threads: int = 16):
        self.zoom = zoom
        self.cache = Path(cache)
        self.mem: dict[tuple, np.ndarray] = {}
        self.lock = threading.Lock()
        self.threads = threads
        self.session = requests.Session()

    def _tile_xy(self, lat, lon):
        n = 2**self.zoom
        lat = np.clip(lat, -85.0511, 85.0511)
        x = (np.asarray(lon) + 180.0) / 360.0 * n
        lr = np.radians(lat)
        y = (1.0 - np.log(np.tan(lr) + 1 / np.cos(lr)) / np.pi) / 2.0 * n
        return x, y

    def _fetch(self, key):
        z, x, y = key
        p = self.cache / str(z) / str(x) / f"{y}.png"
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(4):
                try:
                    r = self.session.get(URL.format(z=z, x=x, y=y), timeout=30)
                    r.raise_for_status()
                    break
                except requests.RequestException:
                    if attempt == 3:
                        raise
            tmp = p.with_suffix(".tmp")
            tmp.write_bytes(r.content)
            tmp.rename(p)
        a = np.asarray(Image.open(p).convert("RGB"), dtype=np.float32)
        e = a[..., 0] * 256 + a[..., 1] + a[..., 2] / 256 - 32768
        return np.maximum(e, 0.0)

    def _ensure(self, keys):
        need = [k for k in keys if k not in self.mem]
        if not need:
            return
        with ThreadPoolExecutor(self.threads) as ex:
            for k, arr in zip(need, ex.map(self._fetch, need)):
                self.mem[k] = arr

    def elevation(self, lat, lon):
        """Bilinear orthometric elevation (m) at arbitrary points."""
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)
        shape = lat.shape
        lat = lat.ravel()
        lon = lon.ravel()
        X, Y = self._tile_xy(lat, lon)
        px = X * 256 - 0.5
        py = Y * 256 - 0.5
        x0 = np.floor(px).astype(np.int64)
        y0 = np.floor(py).astype(np.int64)
        fx = px - x0
        fy = py - y0
        out = np.zeros(len(lat))
        # gather the four neighbours, which may sit in different tiles
        corners = [(0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)]
        n = 2**self.zoom
        keys_all = set()
        idx = []
        for dx, dy, w in corners:
            gx = x0 + dx
            gy = np.clip(y0 + dy, 0, n * 256 - 1)
            tx = (gx // 256) % n
            ty = gy // 256
            idx.append((tx, ty, gx % 256, gy % 256, w))
            keys_all.update(zip(tx.tolist(), ty.tolist()))
        self._ensure([(self.zoom, a, b) for a, b in keys_all])
        for tx, ty, ix, iy, w in idx:
            key = tx * n + ty
            for k in np.unique(key):
                m = key == k
                arr = self.mem[(self.zoom, int(k // n), int(k % n))]
                out[m] += arr[iy[m], ix[m]] * w[m]
        return out.reshape(shape)
