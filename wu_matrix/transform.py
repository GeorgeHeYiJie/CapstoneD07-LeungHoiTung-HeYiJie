"""Wu et al., 2023 (TimesNet) 1D-to-2D matrix conversion.

A battery table is a multivariate series X with shape [T, C]: one row per
cycle, one column per feature. The conversion is:

1. FFT along the cycle axis. Average the amplitude over the C features.
2. Ignore the zero-frequency bin. Keep the k strongest frequencies f.
3. Period length p = T // f. This is the integer division used by the
   TimesNet implementation. The paper text writes ceil(T / f); for these
   lengths the two differ only when f does not divide T, and the working
   reshape uses the integer quotient.
4. Zero-pad the end of the series until its length is a multiple of p.
5. Reshape to [n_periods, p, C].

Rows are successive periods (inter-period variation). Columns are the phase
inside one period (intra-period variation). The same grid is used for every
feature. Each selected period produces its own matrix, because the period
lengths differ and cannot sit in one rectangle.
"""

from __future__ import annotations

import numpy as np

def dominant_periods(series: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return frequency indices, period lengths and amplitudes.

    ``series`` has shape [T, C]. Frequencies are shared across features.
    """
    if series.ndim != 2:
        raise ValueError(f"expected shape [cycles, features], got {series.shape}")
    n_cycles = series.shape[0]
    if n_cycles < 2:
        raise ValueError("need at least 2 cycles to estimate a period")
    if k < 1:
        raise ValueError("k must be at least 1")
    if not np.isfinite(series).all():
        raise ValueError("series contains NaN or infinity")

    spectrum = np.fft.rfft(series, axis=0)
    amplitude = np.abs(spectrum).mean(axis=1).copy()
    amplitude[0] = 0.0
    k_eff = min(k, amplitude.shape[0] - 1)
    # Strongest bins, then sorted from largest amplitude to smallest.
    candidates = np.argpartition(amplitude, -k_eff)[-k_eff:]
    order = np.argsort(amplitude[candidates])[::-1]
    frequencies = candidates[order].astype(np.int64)
    frequencies = np.maximum(frequencies, 1)
    periods = np.maximum(n_cycles // frequencies, 1)
    periods = np.minimum(periods, n_cycles).astype(np.int64)
    return frequencies, periods, amplitude[frequencies].astype(np.float64)


def reshape_period(series: np.ndarray, period: int) -> np.ndarray:
    """Pad ``series`` and reshape one period to [n_periods, period, features]."""
    if series.ndim != 2:
        raise ValueError(f"expected shape [cycles, features], got {series.shape}")
    n_cycles = series.shape[0]
    if period < 1 or period > n_cycles:
        raise ValueError(f"period {period} is outside 1..{n_cycles}")
    remainder = n_cycles % period
    if remainder == 0:
        padded = series
    else:
        pad = period - remainder
        padded = np.pad(series, ((0, pad), (0, 0)), mode="constant", constant_values=0.0)
    n_periods = padded.shape[0] // period
    return padded.reshape(n_periods, period, series.shape[1])


def series_to_matrices(series: np.ndarray, k: int = 5) -> dict:
    """Convert one [T, C] series into k period matrices.

    The returned dict contains:

    - ``matrices``: list of float64 arrays, each [n_periods, period, C]
    - ``periods``: int64 array [k_eff]
    - ``frequencies``: int64 array [k_eff], FFT bin indices
    - ``amplitudes``: float64 array [k_eff]
    - ``n_cycles``: original T, before zero padding
    """
    values = np.asarray(series, dtype=np.float64)
    frequencies, periods, amplitudes = dominant_periods(values, k)
    matrices = [reshape_period(values, int(period)) for period in periods]
    return {
        "matrices": matrices,
        "periods": periods,
        "frequencies": frequencies,
        "amplitudes": amplitudes,
        "n_cycles": int(values.shape[0]),
        "n_features": int(values.shape[1]),
    }
