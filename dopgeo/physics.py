"""Doppler models.

Two models:
* straight_line: constant-velocity, straight flight past the observer. Used to *measure*
  an event from the spectrogram (speed, closest distance, closest-approach time, rest frequency).
* Trajectory.doppler_factor: the exact curve an observer would hear from a real ADS-B
  trajectory, using the position/velocity at the *emission* time of each heard sound.

In both, the frequency heard at observer time t is f0 * c / (c + v_r(tau)), where tau is the
emission time solving t = tau + |p(tau) - o| / c and v_r is the source velocity component
pointing away from the observer.
"""
from __future__ import annotations

import numpy as np

from .geo import ecef, enu_to_ecef_vec


def speed_of_sound(temp_c: float = 20.0) -> float:
    return 331.3 * np.sqrt(1 + temp_c / 273.15)


def straight_line(t, t_heard, f0, v, d, c):
    """Frequency heard at observer times t for a straight constant-speed pass.

    t_heard: observer time at which the closest-approach sound arrives (frequency == f0 there).
    v: speed (m/s), d: closest slant distance (m).
    """
    T = np.asarray(t, float) - (t_heard - d / c)  # time since the closest-approach emission
    a = c * c - v * v
    disc = c**4 * T * T - a * (c * c * T * T - d * d)
    s = (c * c * T - np.sqrt(np.maximum(disc, 0.0))) / a  # emission time relative to CPA emission
    x = v * s
    r = np.sqrt(d * d + x * x)
    vr = v * x / r
    return f0 * c / (c + vr)


class Trajectory:
    """ECEF trajectory of one flight from ADS-B points.

    Position: linear between reported points (readsb stores a point whenever the track changes,
    so straight pieces are long and turns are densely sampled). Velocity: the aircraft's own
    reported ground speed / track / vertical rate, interpolated linearly; finite differences only
    where it is missing. Keeping them separate avoids the overshoot loops a single spline makes
    when reported velocity and position samples disagree slightly.
    """

    def __init__(self, t, lat, lon, h_ellipsoid, gs, track, vrate):
        t = np.asarray(t, float)
        keep = np.r_[True, np.diff(t) > 0.05]
        t, lat, lon, h = t[keep], np.asarray(lat)[keep], np.asarray(lon)[keep], np.asarray(h_ellipsoid)[keep]
        gs = np.asarray(gs, float)[keep]
        track = np.asarray(track, float)[keep]
        vrate = np.asarray(vrate, float)[keep]
        pos = ecef(lat, lon, h)
        fd = np.gradient(pos, t, axis=0) if len(t) > 1 else np.zeros_like(pos)
        trk = np.radians(track)
        vel = enu_to_ecef_vec(lat, lon, gs * np.sin(trk), gs * np.cos(trk), np.where(np.isfinite(vrate), vrate, 0.0))
        bad = ~np.isfinite(vel).all(axis=1)
        vel[bad] = fd[bad]
        self.t, self.P, self.V = t, pos, vel
        self.t_min, self.t_max = float(t[0]), float(t[-1])

    def _interp(self, A, tau):
        tau = np.asarray(tau, float)
        out = np.empty(tau.shape + (3,))
        for k in range(3):
            out[..., k] = np.interp(tau, self.t, A[:, k])
        return out

    def pos(self, tau):
        return self._interp(self.P, tau)

    def vel(self, tau):
        return self._interp(self.V, tau)

    def emission_time(self, obs, t_obs, c, iters=6):
        """obs (..., 3) ECEF observer; t_obs (..., n) observer times -> emission times (..., n)."""
        tau = np.array(t_obs, float, copy=True)
        o = obs[..., None, :]
        for _ in range(iters):
            r = np.linalg.norm(self.pos(tau) - o, axis=-1)
            tau = t_obs - r / c
        return tau

    def doppler_factor(self, obs, t_obs, c):
        """g(t) such that heard frequency = f0 * g(t). Also returns slant range (m) and validity."""
        tau = self.emission_time(obs, t_obs, c)
        d = self.pos(tau) - obs[..., None, :]
        r = np.linalg.norm(d, axis=-1)
        vr = np.einsum("...k,...k->...", self.vel(tau), d) / r
        g = c / (c + vr)
        valid = (tau >= self.t_min) & (tau <= self.t_max)
        return g, r, valid

    def cpa(self, obs, c, t_lo=None, t_hi=None, step=0.5):
        """Closest approach to each observer: (t_emission, t_heard, slant_distance, speed)."""
        t_lo = self.t_min if t_lo is None else max(t_lo, self.t_min)
        t_hi = self.t_max if t_hi is None else min(t_hi, self.t_max)
        ts = np.arange(t_lo, t_hi + step, step)
        P = self.pos(ts)
        obs = np.atleast_2d(obs)
        out = np.empty((len(obs), 4))
        for i, o in enumerate(obs):
            r = np.linalg.norm(P - o, axis=1)
            k = int(np.argmin(r))
            out[i] = ts[k], ts[k] + r[k] / c, r[k], np.linalg.norm(self.vel(ts[k]))
        return out
