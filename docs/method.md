# How doppler-geo works

This document explains the method step by step: what is measured in the audio, which physics links it
to an aircraft, and how the search over a whole day and a large area is kept fast. The code references
are to modules in `dopgeo/`.

## 1. What the audio contains

Aircraft engines and propellers produce steady tones, typically 50 to 300 Hz for propeller blade
passes and engine harmonics. A microphone hears them shifted by the Doppler effect:

    f_heard(t) = f_rest * c / (c + rdot(tau))

where `c` is the speed of sound, `rdot` the rate at which the aircraft's distance to the microphone
changes, and `tau` the time the sound was emitted (it arrives `R / c` later, `R` being the distance).
While the aircraft approaches, `rdot < 0` and the tone is higher; once it recedes, the tone is lower.
These tones are often inaudible to a person listening casually, because they are quiet and low, but
they are clearly visible in a spectrogram.

`audio.py` decodes any audio or video file with ffmpeg to mono at 2 kHz and computes a whitened
spectrogram (2 s windows, 0.25 s hop). `measure.py` links spectral peaks into continuous tone lines,
joins pieces of one line that broke where it crossed another line or faded for a few seconds (the later
piece must start where the earlier one's slope leads, within a few seconds), and keeps the stretches
where the frequency falls: these are the passes. Tones of one aircraft (harmonics, or an engine and a
propeller at any frequency ratio) share the same Doppler factor, so their log frequencies keep a
constant difference while they fall; such lines are grouped and count as one pass. Each pass is fitted with a
straight constant-speed pass model (giving a rough speed, closest distance and closest-approach time),
and labelled *regular* if that model fits, *irregular* otherwise, and *clear* if the line is strong and
long enough to search with.

## 2. The feature that ties a pass to a flight: the fastest fall

The rest frequency `f_rest` of an aircraft is unknown, and it trades off against timing when fitting
a curve. Only changes of log frequency are free of it. For every clear pass, `peaks.py` computes the
fall rate of log frequency (5 s moving average, then a 4 s difference) and finds where it peaks:

* `t_k`: the audio time at which the tone falls fastest,
* `D_k`: how fast it falls there (per second).

A pass whose line shows this maximum with the fall slowing on both sides is said to have a *clear
fastest fall*. About 80 % of clear passes do.

For a listener at distance `R`, an aircraft with velocity `v` and acceleration `a` produces the fall rate

    D = [ kappa * (|v|^2 - rdot^2) / R + a . u ] * c / (c + rdot)^2

with `u` the unit vector from listener to aircraft and `kappa` the flattening of very sharp peaks by the
5 s smoothing (`kin.py`). Two consequences matter:

* For a straight, constant-speed pass, the fastest fall is heard at the closest approach, with
  `D = kappa * v^2 / (c d)`. Speed and distance are linked, but the time is exact.
* For a turning, climbing or accelerating aircraft, the `a . u` term moves the fastest fall away from
  the closest approach, sometimes by more than a minute, and changes its height. A model of a straight
  pass gets the time wrong there; the formula above does not, because `a` comes from the flight's own
  ADS-B velocities.

Departures and arrivals near airports are exactly the second case, so this matters most in the
busiest airspace.

## 3. The search

The unknowns are the place (anywhere in the search area) and the recording start time `T0` (anywhere
in the allowed window, typically a whole day). A candidate is right if many passes `k` each have some
flight that, heard from the place, has its fastest fall at time `T0 + t_k` with height `D_k`.

### 3.1 Peak rows: independent of the audio

For every flight and every 1 km cell within 15 km of its track, `kin.py` scans the flight's 1 s samples
around its closest approach to the cell centre and finds the peaks of `D`. Each peak is stored as a row:

* heard time and rate at the cell centre,
* how both change per metre east and north (from four nearby evaluations).

Time and place are tied for a moving aircraft: standing 500 m further along its track delays the pass
by about `500 / v` seconds. A plain grid of cells and time bins cannot express that and has to allow
the whole spread (10 to 25 s per cell), which in busy airspace lets almost any cell match almost any
pass. With the gradients, a row is exact at any of 25 points 200 m apart inside the cell, at the cost
of a single 1 km evaluation.

Two refinements keep this both accurate and affordable:

* Passes closer than one cell change too fast across it to be linearised; they get exact rows at each
  of the 25 points instead.
* Passes further than 4 km change slowly; one evaluation at the centre of each 2 x 2 block of cells
  serves all four, carried to each by the gradients.

None of this depends on the recording. It is the expensive part of the search, and could be computed
once per day and area and stored.

### 3.2 The join: depends on the audio

Each pass selects the rows whose rate fits it (within 30 %, plus the row's own spread) and converts
their heard time into a start time `T0 = heard time - t_k`. A candidate is a point and start time where
at least `m` passes fit with one common start time (by default 60 % of the passes with a clear fastest
fall).

If a candidate needs `m` of `K` passes, it must contain at least one of any `K - m + 1` passes. The
search therefore starts only from the matches of the `K - m + 1` most precisely timed passes (the
*anchors*). For each anchor match in a cell, the other passes are looked up in its time window: first
over the whole cell (exact for linear times, cheap), and only if enough passes could agree, at the 25
points. Agreeing passes must share one common start time, not merely each overlap the anchor. Cells
are independent, so they are processed in parallel.

### 3.3 Checking with the matched flights

In busy airspace the join finds millions of coincidences, and in real recordings the measured fastest fall
is noisy enough that the true place does not always have the most votes. Every coincidence already names a
flight for each pass, so up to three million of them are checked with the exact Doppler curves of those
flights only (one flight per pass, 16 line points, shifts of +/- 6 s in 3 s steps); three million take
about 8 seconds on 8 cores.
The best 6000 by this check go on.

### 3.4 Verification with exact Doppler curves

These candidates are scored by `fine.py`: for every pass, every flight within 12 km of
the candidate is tried; the exact Doppler curve it would produce there (emission time solved, positions
interpolated from ADS-B, ground height from a terrain model) is fitted to the tracked line, with the
rest frequency solved in closed form. Per pass the best flight counts, and all passes must agree on one
start time (shifts of +/- 6 s are searched). The best places are then refined on a 100 m and finally a
25 m grid.

Passes without a clear fastest fall still take part here. When fewer than three passes show a clear
fastest fall, the candidate generation falls back to an older method (`coarse.py`) that votes with
straight-pass fits; it is accurate away from airports but weaker near them.

### 3.5 Confidence

`high`: at least 4 passes explained (fit >= 0.5), a lead of at least 1.5 passes over the best place at
least 2 km away, and a total score of at least 60 % of the passes. `medium`: 3 explained, a lead of 0.75,
40 %. Otherwise `low`. A `low` answer with few passes means the recording does not contain enough
information to rule out other places, not necessarily that the answer is wrong.

## 4. Data

* **ADS-B history**: the [adsb.lol globe_history](https://github.com/adsblol) daily releases (ODbL 1.0).
  One UTC day is about 3.5 GB to download and about 1 GB once extracted to the compact table used
  here. Only aircraft below 15,000 ft are kept.
* **Terrain**: [Terrain Tiles](https://registry.opendata.aws/terrain-tiles/) on AWS, for the ground
  height of each candidate place (the listener's height enters the slant distance).
* **Aircraft types**: the ICAO type table of [tar1090-db](https://github.com/wiedehopf/tar1090-db), to
  tell jets from propeller aircraft and helicopters (`exclude_jets`).
* **Geoid**: EGM96 through PROJ, to convert between ellipsoidal (ADS-B geometric) and terrain heights.

## 5. Limits

* Every pass must come from an aircraft that broadcast ADS-B and was received by the network. Aircraft
  without ADS-B, or flying where coverage is poor, cannot explain a pass.
* With few passes (two or three), many places can explain them equally well, especially over large
  areas; the answer is then marked `low`.
* The search area costs time roughly in proportion to the traffic in it. A metropolitan area (160 km
  square) takes from 10 s to about 2 minutes on a laptop; the whole contiguous US would take hours.
* Tracked lines in real recordings can come from things other than aircraft (road traffic, machinery,
  boats). Such lines can only add noise to the search.
