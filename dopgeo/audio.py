"""Audio loading and whitened spectrograms on which Doppler tones are measured and scored."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass

import numpy as np
from scipy.signal import stft

DEFAULT_SR = 2000  # Hz. Propeller fundamentals and the first harmonics sit well below 1 kHz.


def load(path, sr=DEFAULT_SR, start=None, duration=None) -> np.ndarray:
    """Decode any audio/video file to mono float32 at `sr` with ffmpeg."""
    cmd = ["ffmpeg", "-v", "error"]
    if start is not None:
        cmd += ["-ss", str(start)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, check=True, capture_output=True).stdout
    return np.frombuffer(raw, np.float32).copy()


@dataclass
class Spectrogram:
    """Whitened log-power image. values[f_index, t_index]; regular axes."""

    values: np.ndarray
    t0: float  # audio time of column 0 (s)
    dt: float
    f0: float  # frequency of row 0 (Hz)
    df: float

    @property
    def times(self):
        return self.t0 + self.dt * np.arange(self.values.shape[1])

    @property
    def freqs(self):
        return self.f0 + self.df * np.arange(self.values.shape[0])

    def crop(self, t_lo, t_hi, f_lo, f_hi) -> "Spectrogram":
        ti = slice(max(0, int(np.floor((t_lo - self.t0) / self.dt))), int(np.ceil((t_hi - self.t0) / self.dt)) + 1)
        fi = slice(max(0, int(np.floor((f_lo - self.f0) / self.df))), int(np.ceil((f_hi - self.f0) / self.df)) + 1)
        return Spectrogram(
            self.values[fi, ti],
            self.t0 + (ti.start or 0) * self.dt,
            self.dt,
            self.f0 + (fi.start or 0) * self.df,
            self.df,
        )

    def sample(self, t, f):
        """Bilinear lookup; zero outside the image."""
        v = self.values
        x = (np.asarray(t) - self.t0) / self.dt
        y = (np.asarray(f) - self.f0) / self.df
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        fx = x - x0
        fy = y - y0
        ok = (x0 >= 0) & (y0 >= 0) & (x0 < v.shape[1] - 1) & (y0 < v.shape[0] - 1)
        x0c = np.where(ok, x0, 0)
        y0c = np.where(ok, y0, 0)
        out = (
            v[y0c, x0c] * (1 - fx) * (1 - fy)
            + v[y0c, x0c + 1] * fx * (1 - fy)
            + v[y0c + 1, x0c] * (1 - fx) * fy
            + v[y0c + 1, x0c + 1] * fx * fy
        )
        return np.where(ok, out, 0.0)


def spectrogram(x, sr, t_offset=0.0, win_s=2.0, hop_s=0.25, f_lo=20.0, f_hi=600.0, pad=2) -> Spectrogram:
    """Whitened spectrogram.

    Long windows (2 s -> 0.5 Hz bins, zero-padded) because aircraft tones are narrow and slow.
    Whitening: per-frequency median removed and divided by a robust spread, then each column's
    median removed, so steady hums and broadband bursts both fall to ~0 and tonal lines stand out.
    """
    nper = int(win_s * sr)
    hop = int(hop_s * sr)
    f, t, Z = stft(x, sr, window="hann", nperseg=nper, noverlap=nper - hop, nfft=nper * pad, boundary=None, padded=False)
    P = 10 * np.log10(np.abs(Z) ** 2 + 1e-20)
    m = (f >= f_lo) & (f <= f_hi)
    P = P[m]
    f = f[m]
    med = np.median(P, axis=1, keepdims=True)
    mad = np.median(np.abs(P - med), axis=1, keepdims=True) * 1.4826 + 1e-6
    W = (P - med) / mad
    W -= np.median(W, axis=0, keepdims=True)
    W = np.clip(W, 0, 8).astype(np.float32)
    # stft time stamps are window centres
    return Spectrogram(W, t_offset + float(t[0]), float(t[1] - t[0]), float(f[0]), float(f[1] - f[0]))
