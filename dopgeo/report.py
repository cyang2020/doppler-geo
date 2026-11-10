"""Human-readable outputs: map (Leaflet HTML), curve overlays, summary."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import score


def write(out: Path, top, ranked, refined, events, spec, bank, dem, c):
    _plots(out, top, events, spec, bank, dem, c)
    _map(out, top, ranked, refined)


def _plots(out, top, events, spec, bank, dem, c):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obs = score.observers([top["lat"]], [top["lon"]], dem)[0]
    T0 = np.datetime64(top["recording_start_utc"].rstrip("Z")).astype("datetime64[ms]").astype(np.int64) / 1000
    fig, axes = plt.subplots(len(events), 1, figsize=(14, 4.2 * len(events)), squeeze=False)
    for ax, e, fl in zip(axes[:, 0], events, top["flights"]):
        lo, hi = e.window[0] - 15, e.window[1] + 15
        f_lo = max(spec.f0, min(e.band[0], e.f0 * 0.6))
        f_hi = max(e.band[1], e.f0 * max(e.harmonics) * 1.4)
        sub = spec.crop(lo, hi, f_lo, f_hi)
        ax.pcolormesh(sub.times, sub.freqs, sub.values, vmin=0, vmax=6, cmap="magma", shading="auto")
        i = int(np.searchsorted(bank.fids, fl["fid"]))
        if fl["fid"] < 0 or i >= len(bank.fids) or bank.fids[i] != fl["fid"]:
            ax.set_title(f"pass at {e.t_heard:.0f}s: no flight")
            continue
        a, b = bank.start[i], bank.start[i + 1]
        t = np.arange(lo, hi, 0.25)
        g = np.empty(len(t))
        ok = np.zeros(len(t), bool)
        score._curve(bank.t, bank.pos, bank.vel, a, b, obs, T0 + t[0], 0.25, len(t), c, g, ok)
        la, lf, lm = score.line_grid(e, 0.25)
        f0 = np.nanmedian(np.asarray(e.line_f) / np.interp(e.line_t, t, np.where(ok, g, np.nan)))
        for h in e.harmonics:
            ax.plot(t[ok], f0 * h * g[ok], "c--", lw=1, alpha=0.8)
        ax.set_title(f"pass at {e.t_heard:.0f}s  vs  {fl['type'] or '?'} {fl['reg'] or fl['icao']}  (model curve dashed)")
        ax.set_xlabel("audio time (s)")
        ax.set_ylabel("Hz")
    plt.tight_layout()
    plt.savefig(out / "curves.png", dpi=80)
    plt.close(fig)


def _map(out, top, ranked, refined):
    cands = ranked.head(50)[["lat", "lon", "score"]].round(6).to_dict("records")
    heat = refined[0]["heat"].round(6).to_dict("records") if refined else []
    html = """<!doctype html><html><head><meta charset="utf-8"><title>dopgeo result</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>html,body,#m{height:100%;margin:0}#info{position:absolute;z-index:999;top:10px;right:10px;background:#fff;padding:8px 12px;font:13px sans-serif;max-width:360px;box-shadow:0 1px 4px #0005}</style>
</head><body><div id="m"></div><div id="info"></div><script>
const top=__TOP__, cands=__CANDS__, heat=__HEAT__;
const m=L.map('m').setView([top.lat,top.lon],14);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{maxZoom:19,attribution:'OSM'}).addTo(m);
const hmax=Math.max(...heat.map(h=>h.score)), hmin=Math.min(...heat.map(h=>h.score));
heat.forEach(h=>{const v=(h.score-hmin)/(hmax-hmin+1e-9); if(v>0.5) L.circle([h.lat,h.lon],{radius:12,stroke:false,fillColor:'#e4572e',fillOpacity:0.08+0.6*Math.pow(v,4)}).addTo(m)});
cands.forEach((c,i)=>L.circleMarker([c.lat,c.lon],{radius:i?4:7,color:i?'#555':'#000',fillColor:i?'#aaa':'#ffd400',fillOpacity:1}).bindPopup('#'+(i+1)+' score '+c.score.toFixed(2)).addTo(m));
L.marker([top.lat,top.lon]).addTo(m);
document.getElementById('info').innerHTML='<b>Best location</b><br>'+top.lat.toFixed(5)+', '+top.lon.toFixed(5)+'<br>score '+top.score.toFixed(2)+(top.margin!=null?' (margin '+top.margin.toFixed(2)+')':'')+'<br>recording start '+top.recording_start_utc+'<br>flights: '+top.flights.map(f=>(f.type||'?')+' '+(f.reg||f.icao)).join(', ');
</script></body></html>"""
    html = html.replace("__TOP__", json.dumps(top, default=float)).replace("__CANDS__", json.dumps(cands)).replace("__HEAT__", json.dumps(heat))
    (out / "map.html").write_text(html)
