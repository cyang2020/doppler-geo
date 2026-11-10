"""Map of a result: the found place, the flights that explain the passes, and optionally the truth.

    from dopgeo.plot import plot_result
    plot_result(result, "result.png", truth=(lat, lon))

A map background (USGS National Map tiles, public domain) is drawn when the optional `contextily`
package is installed and the tile server can be reached (pip install 'doppler-geo[plot]'); otherwise
the axes are plain.
"""
from __future__ import annotations

import numpy as np
import pyproj

_TO_WM = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def _wm(lat, lon):
    x, y = _TO_WM.transform(np.asarray(lon, float), np.asarray(lat, float))
    return np.asarray(x), np.asarray(y)


def _basemap(ax, zoom=None):
    """USGS National Map tiles (public domain, US only); Esri's grey canvas elsewhere or if USGS fails."""
    try:
        import contextily as ctx
    except ImportError:
        return False
    kw = {} if zoom is None else dict(zoom=zoom)
    for src, attr in ((ctx.providers.USGS.USTopo, "USGS The National Map"),
                      (ctx.providers.Esri.WorldGrayCanvas, "Esri, HERE, Garmin, (c) OpenStreetMap contributors")):
        try:
            ctx.add_basemap(ax, source=src, crs="EPSG:3857", attribution=attr, attribution_size=6, alpha=0.8, **kw)
            return True
        except Exception:
            continue
    return False


def plot_result(result, path=None, truth=None, title=None, span_km=14.0, zoom_m=900.0, basemap=True):
    """Two panels: the matched flights' tracks around the found place (span_km wide), and a close-up
    (zoom_m half-width) with the truth if given. Returns the matplotlib figure; saved to path if given."""
    import matplotlib

    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Web Mercator stretches distances by 1/cos(lat); scale the view so spans are true kilometres
    k = 1.0 / np.cos(np.radians(result.lat))
    fx, fy = _wm(result.lat, result.lon)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 7.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
    h1 = 0.5 * span_km * 1000 * k
    cmap = plt.get_cmap("tab10")
    shown = 0
    labels = []
    for i, m in enumerate(sorted(result.matches, key=lambda m: m.audio_time_s)):
        tr = result.tracks.get(str(m.flight_id))
        if tr is None or m.fit < 0.5:
            continue
        x, y = _wm(tr["lat"], tr["lon"])
        col = cmap(shown % 10)
        a1.plot(x, y, color=col, lw=1.6, alpha=0.9)
        # label where the track is inside the view and clear of earlier labels, nearest the found place first
        inside = np.flatnonzero((np.abs(x - fx) < 0.92 * h1) & (np.abs(y - fy) < 0.92 * h1))
        if len(inside) == 0:
            shown += 1
            continue
        cand = inside[np.argsort(np.hypot(x[inside] - fx, y[inside] - fy))]
        j = int(cand[0])
        for c_ in cand:
            if all(np.hypot(x[c_] - px_, y[c_] - py_) > 0.09 * h1 for px_, py_ in labels):
                j = int(c_)
                break
        labels.append((x[j], y[j]))
        a1.annotate(f"{m.audio_time_s:.0f} s  {m.aircraft_type or '?'} {m.registration}", (x[j], y[j]), fontsize=7.5,
                    color=col, xytext=(4, 4), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
        shown += 1
    a1.plot(fx, fy, marker="*", ms=17, color="crimson", mec="white", mew=1.2, zorder=6, label="found")
    if result.runner_up is not None:
        rx, ry = _wm(result.runner_up.lat, result.runner_up.lon)
        if abs(rx - fx) < h1 and abs(ry - fy) < h1:
            a1.plot(rx, ry, marker="X", ms=10, color="dimgray", mec="white", zorder=5, label="runner-up place")
    if truth is not None:
        tx, ty = _wm(*truth)
        a1.plot(tx, ty, marker="o", ms=9, mfc="none", mec="green", mew=2.2, zorder=7, label="truth")
    a1.set_xlim(fx - h1, fx + h1)
    a1.set_ylim(fy - h1, fy + h1)
    a1.set_title(f"{shown} flights explaining the passes (tracks during the recording)", fontsize=10)
    a1.legend(loc="lower left", fontsize=8)

    h2 = zoom_m * k
    a2.plot(fx, fy, marker="*", ms=20, color="crimson", mec="white", mew=1.2, zorder=6, label="found")
    a2.add_patch(plt.Circle((fx, fy), max(result.spread_m, 25.0) * k, fill=False, ls="--", color="crimson", lw=1.2,
                            label=f"places within a quarter pass of the best (r = {result.spread_m:.0f} m)"))
    sub = f"start {result.recording_start:%Y-%m-%d %H:%M:%S} UTC, confidence {result.confidence}"
    if truth is not None:
        a2.plot(tx, ty, marker="o", ms=12, mfc="none", mec="green", mew=2.4, zorder=7, label="truth")
        err = np.hypot((result.lat - truth[0]) * 111000,
                       (result.lon - truth[1]) * 111000 * np.cos(np.radians(truth[0])))
        sub = f"error {err:.0f} m, " + sub
    a2.set_xlim(fx - h2, fx + h2)
    a2.set_ylim(fy - h2, fy + h2)
    a2.set_title(sub, fontsize=10)
    a2.legend(loc="lower left", fontsize=8)
    for ax in (a1, a2):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    if basemap:
        _basemap(a1)
        _basemap(a2)
    fig.suptitle(title or f"{result.lat:.5f}, {result.lon:.5f}", fontsize=12)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig
