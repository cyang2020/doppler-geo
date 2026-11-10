"""HTTP API (optional: pip install 'doppler-geo[server]'; run with `dopgeo serve`).

    POST /locate   multipart form: file=<audio>, plus the same options as dopgeo.locate():
                   date & tz, or start & start_tolerance_s; center_lat, center_lon & radius_km, or
                   lat_min, lat_max, lon_min, lon_max; exclude_jets; max_passes.
                   Returns the Result as JSON.
    POST /passes   multipart form: file=<audio>. Returns the passes found in it.
    GET  /health

A search takes from seconds to a minute or more and holds its worker for that time; this server is meant
for local or internal use, not as a public endpoint.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from . import __version__, api

app = FastAPI(title="doppler-geo", version=__version__,
              description="Locate a recording from aircraft Doppler tones and public ADS-B history.")


def _save(upload: UploadFile, tmp: str) -> Path:
    p = Path(tmp) / (Path(upload.filename or "audio").name or "audio")
    p.write_bytes(upload.file.read())
    return p


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.post("/passes")
def passes(file: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as tmp:
        ps = api.detect_passes(_save(file, tmp))
    return [dict(vars(p), line_t=None, line_f=None) for p in ps]


@app.post("/locate")
def locate(file: UploadFile = File(...), date: str | None = Form(None), tz: str = Form("UTC"),
           start: str | None = Form(None), start_tolerance_s: float = Form(600.0),
           center_lat: float | None = Form(None), center_lon: float | None = Form(None), radius_km: float = Form(50.0),
           lat_min: float | None = Form(None), lat_max: float | None = Form(None), lon_min: float | None = Form(None),
           lon_max: float | None = Form(None), exclude_jets: bool = Form(False), max_passes: int = Form(12),
           include_tracks: bool = Form(False)):
    if lat_min is not None and None not in (lat_max, lon_min, lon_max):
        area = api.Area(lat_min, lat_max, lon_min, lon_max)
    elif center_lat is not None and center_lon is not None:
        area = api.Area.circle(center_lat, center_lon, radius_km)
    else:
        raise HTTPException(422, "give center_lat & center_lon (with radius_km), or lat_min/lat_max/lon_min/lon_max")
    if date is None and start is None:
        raise HTTPException(422, "give date (with tz) or start (with start_tolerance_s)")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            res = api.locate(_save(file, tmp), date=date, tz=tz, start=start, start_tolerance_s=start_tolerance_s,
                             area=area, exclude_jets=exclude_jets, max_passes=max_passes)
        except RuntimeError as e:
            raise HTTPException(422, str(e)) from e
    d = res.to_dict()
    if not include_tracks:
        d.pop("tracks")
    return d
