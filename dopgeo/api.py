"""Public API.

    import dopgeo

    result = dopgeo.locate("recording.wav", date="2026-09-08", tz="America/New_York",
                           center=(40.8, -73.4), radius_km=80)
    print(result.lat, result.lon, result.recording_start, result.confidence)

Everything here is a thin layer over the search in pipeline.py: it turns friendly arguments (a date and
time zone, a centre and radius) into the search area and start-time window, and the result into typed
objects.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import adsb, audio, measure, peaks, pipeline
from .config import adsb_dir

CONUS = (24.0, 49.6, -125.0, -66.5)


# --------------------------------------------------------------------------- inputs

@dataclass(frozen=True)
class Area:
    """A search area as a latitude / longitude box (degrees)."""

    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    @classmethod
    def circle(cls, lat: float, lon: float, radius_km: float) -> Area:
        """The box around a circle (the search covers the whole box)."""
        d = radius_km / 111.0
        e = d / np.cos(np.radians(lat))
        return cls(lat - d, lat + d, lon - e, lon + e)

    @classmethod
    def conus(cls) -> Area:
        return cls(*CONUS)

    def bbox(self) -> tuple:
        return (self.lat_min, self.lat_max, self.lon_min, self.lon_max)

    def area_km2(self) -> float:
        h = (self.lat_max - self.lat_min) * 111.0
        w = (self.lon_max - self.lon_min) * 111.0 * np.cos(np.radians(0.5 * (self.lat_min + self.lat_max)))
        return float(h * w)


@dataclass(frozen=True)
class TimeWindow:
    """When the recording may have started (UTC epoch seconds, inclusive)."""

    start: float
    end: float

    @classmethod
    def day(cls, date: str, tz: str = "UTC", duration_s: float = 0.0) -> TimeWindow:
        """Any start on a local calendar day, such that the recording also ends on it."""
        d = pd.Timestamp(date).tz_localize(tz).tz_convert("UTC").timestamp()
        return cls(d, d + 86400.0 - duration_s)

    @classmethod
    def around(cls, start: str | float | datetime, tolerance_s: float = 600.0) -> TimeWindow:
        """A start known roughly: start +/- tolerance_s. Strings without an offset are read as UTC."""
        t = _epoch(start)
        return cls(t - tolerance_s, t + tolerance_s)


# --------------------------------------------------------------------------- outputs

@dataclass
class Pass:
    """An aircraft pass found in the audio: a tracked tone whose frequency falls as the aircraft goes by."""

    index: int
    time_s: float  # audio time of the closest approach as fitted by a straight pass
    rest_frequency_hz: float  # the tone's frequency without Doppler shift (fitted)
    clear: bool  # strong and long enough to search with
    straight: bool  # the line fits a straight, constant-speed pass
    fastest_fall_s: float | None  # audio time when the tone falls fastest, if the line shows it clearly
    fastest_fall_rate: float | None  # that fall rate, in log frequency per second
    line_t: list = field(repr=False, default_factory=list)  # the tracked line (audio s, Hz)
    line_f: list = field(repr=False, default_factory=list)


@dataclass
class PassMatch:
    """Which flight explains a pass at the found place and start time."""

    audio_time_s: float
    fit: float  # 0..1, how well the flight's exact Doppler curve matches the tracked line
    flight_id: int  # internal id in the ADS-B tables (-1: none found)
    icao: str
    registration: str
    aircraft_type: str


@dataclass
class Place:
    lat: float
    lon: float
    score: float
    recording_start: datetime


@dataclass
class Result:
    lat: float
    lon: float
    recording_start: datetime  # UTC
    confidence: str  # "high", "medium" or "low"
    score: float  # sum over passes of the fit (0..passes_used)
    passes_used: int
    passes_explained: int  # passes with fit >= 0.5
    spread_m: float  # places within a quarter pass of the best score lie this far from it
    margin: float  # score lead over the best place at least 2 km away
    runner_up: Place | None
    matches: list[PassMatch]
    method: str  # "kinematic" (passes with a clear fastest fall) or "straight-pass"
    audio_duration_s: float
    timing_s: dict
    ranked: list[Place] = field(repr=False, default_factory=list)  # distinct places >= 2 km apart, best first
    tracks: dict = field(repr=False, default_factory=dict)  # flight id -> positions of the matched flights
    output_dir: Path | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["recording_start"] = self.recording_start.isoformat()
        for p in [d["runner_up"]] + d["ranked"]:
            if p is not None:
                p["recording_start"] = p["recording_start"].isoformat()
        d["output_dir"] = None if self.output_dir is None else str(self.output_dir)
        return d

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=float)

    def map_url(self) -> str:
        return f"https://www.openstreetmap.org/?mlat={self.lat:.6f}&mlon={self.lon:.6f}#map=15/{self.lat:.5f}/{self.lon:.5f}"


# --------------------------------------------------------------------------- functions

def detect_passes(audio_file) -> list[Pass]:
    """Aircraft passes in an audio or video file (any format ffmpeg reads)."""
    x = audio.load(audio_file)
    spec = audio.spectrogram(x, audio.DEFAULT_SR)
    out = []
    for i, e in enumerate(measure.scan(spec, log=None)):
        ap = peaks.audio_peak(e)
        out.append(Pass(i, float(e.t_heard), float(e.f0), bool(e.confident), e.note.startswith("regular"),
                        float(ap.t) if ap.ok else None, float(ap.rate) if ap.ok else None,
                        list(map(float, e.line_t)), list(map(float, e.line_f))))
    return out


def locate(audio_file, *, date: str | None = None, tz: str = "UTC", start=None, start_tolerance_s: float = 600.0,
           window: TimeWindow | None = None, center: tuple | None = None, radius_km: float = 50.0,
           area: Area | None = None, exclude_jets: bool = False, temperature_c: float = 20.0,
           max_passes: int = 12, passes: list[int] | None = None, adsb_cache=None, output_dir=None,
           verbose: bool = False) -> Result:
    """Find where and when a recording was made from the aircraft passes in it.

    Time: give `date` (local calendar day, with `tz`), or `start` (+/- `start_tolerance_s`), or `window`.
    Place: give `center` (lat, lon) with `radius_km`, or `area`.
    exclude_jets: ignore known jet types (when the passes are propeller aircraft or helicopters).
    passes: indices from detect_passes() to use instead of choosing automatically.
    output_dir: also write result.json, ranked.csv, map.html and curves.png there.
    ADS-B history for the needed UTC days is downloaded on first use (about 1 GB per day) into
    adsb_cache (default: $DOPGEO_CACHE/adsb or ~/.cache/dopgeo/adsb).
    """
    if isinstance(audio_file, np.ndarray):
        x = audio_file
    else:
        x = audio.load(audio_file)
    duration = len(x) / audio.DEFAULT_SR
    if window is None:
        if date is not None:
            window = TimeWindow.day(date, tz, duration)
        elif start is not None:
            window = TimeWindow.around(start, start_tolerance_s)
        else:
            raise ValueError("give date=, start= or window= (when the recording may have started)")
    if area is None:
        if center is None:
            raise ValueError("give center= (with radius_km=) or area= (where to search)")
        area = Area.circle(center[0], center[1], radius_km)
    log = print if verbose else (lambda *a, **k: None)
    r = pipeline.locate(x, area.bbox(), (window.start, window.end), out_dir=output_dir,
                        adsb_cache=adsb_dir() if adsb_cache is None else adsb_cache, events=passes,
                        n_events=max_passes, temp_c=temperature_c, classes=adsb.NON_JET if exclude_jets else None,
                        log=log)
    return _result(r, output_dir)


def fetch_adsb(date: str, adsb_cache=None) -> Path:
    """Download and extract the adsb.lol history of one UTC day (YYYY-MM-DD); returns its directory."""
    return adsb.ensure_day(date, adsb_dir() if adsb_cache is None else adsb_cache)


def synthesize(lat: float, lon: float, start, duration_s: float, out_wav, *, seed: int = 0, radius_m: float = 6000.0,
               adsb_cache=None) -> dict:
    """A synthetic recording: Doppler-shifted tones of every real flight that passed within radius_m of
    (lat, lon) after `start` (UTC), in noise. Writes out_wav and a .truth.json next to it; returns the truth."""
    from . import synth

    t0 = _epoch(start)
    cache = adsb_dir() if adsb_cache is None else adsb_cache
    days = [adsb.ensure_day(d, cache) for d in pipeline.utc_days(t0 - 600, t0 + duration_s + 600)]
    return synth.synthesize(days, lat, lon, t0, duration_s, out_wav, radius=radius_m, seed=seed, log=lambda *a: None)


# --------------------------------------------------------------------------- helpers

def _epoch(t) -> float:
    if isinstance(t, (int, float)):
        return float(t)
    ts = pd.Timestamp(t)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.timestamp()


def _dt(iso: str) -> datetime:
    return pd.Timestamp(iso.rstrip("Z")).tz_localize("UTC").floor("us").to_pydatetime()


def _result(r: dict, output_dir) -> Result:
    ru = r.get("runner_up") or {}
    ranked = [Place(p["lat"], p["lon"], p["score"], _dt(p["start_utc"])) for p in r.get("ranked", [])]
    runner = ranked[1] if len(ranked) > 1 else None
    if runner is None and ru.get("lat") is not None:
        runner = Place(ru["lat"], ru["lon"], ru["score"], _dt(r["recording_start_utc"]))
    return Result(
        lat=r["lat"], lon=r["lon"], recording_start=_dt(r["recording_start_utc"]), confidence=r["confidence"],
        score=r["score"], passes_used=r["passes_used"], passes_explained=r["passes_explained"], spread_m=r["spread_m"],
        margin=r["margin"], runner_up=runner,
        matches=[PassMatch(f["pass_t"], f["score"], f["fid"], f["icao"] or "", f["reg"] or "", f["type"] or "")
                 for f in r["flights"]],
        method=r["method"], audio_duration_s=r["audio_duration_s"], timing_s=r["timing_s"], ranked=ranked,
        tracks=r.get("tracks", {}), output_dir=None if output_dir is None else Path(output_dir))


__all__ = ["Area", "TimeWindow", "Pass", "PassMatch", "Place", "Result", "detect_passes", "locate", "fetch_adsb",
           "synthesize"]
