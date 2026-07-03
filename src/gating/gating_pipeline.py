"""
IVUS Gating Signal Processing
==============================
Two-signal hybrid algorithm (improved from AIVUS-CAA publication):

Image signal (always available, normal cross-correlation (NCC) of consecutive frames)
    s[n] = 1 - NCC(frame_n, frame_{n+1})  (CCB s0, Talou et al. 2015; abb.: combined correlation and blurring (CCB))
    Bandpass at [0.7, 2.2] x f_heart removes both the slow pullback
    trend (DC) and high-frequency speckle noise.
    Peaks of filtered signal = maximum-motion frames (mid-systole and
    mid-diastole); used as timing landmarks, not stable-phase gating points.

Contour signal  (when ≥50% of frames have drawn contours)
    s[n] = lumen_area[n]  (mm², from report_data)
    Same bandpass filter.
    Peaks of filtered signal = diastole (large, relaxed lumen).
    Troughs = systole (small, compressed lumen).

Heart rate detection
    FFT spectral peak of the correlation signal in the configured range.
    No grid-search optimization -> the FFT estimate is already correct for
    both rest (~60-100 BPM) and stress (~100-210 BPM) imaging.

Notes on Reasoning
------------------
- blur signal, centroid vector, and weight optimization removed:
  they add noise for real IVUS data and degrade the clean correlation signal.
- DT-CWT (Torbati et al. 2019) tested, resulted in more noise than the simple bandpass filter. 
- The 2 x f_heart trick is unnecessary when the lumen area is available:
  it already separates sys and dia at a single f_heart.
- Centroid vector analysis showed SNR = 0.04 (noise-dominated by catheter
  rotation) versus lumen area SNR = 1.54; vector signals not used.
"""

import time
import numpy as np
import pandas as pd
from loguru import logger
from scipy.signal import butter, filtfilt


def _timing(func):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        result = func(*args, **kwargs)
        logger.debug(f"{func.__name__} took {time.time()-t0:.3f}s")
        return result

    return wrapper


@_timing
def prepare_data(
    main_window,
    frames: np.ndarray,
    report_data,
    lower_limit: int = 0,
    x1: int = 50,
    x2: int = 450,
    y1: int = 50,
    y2: int = 450,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare gating signals from IVUS frames and report data.
    If gating data already calculated and config didn't change,
    return cached signals from runtime_data.gating_signal.

    Image signal  : 1 - NCC(frame_n, frame_{n+1}), bandpass filtered.
    Contour signal: lumen_area from report_data, bandpass filtered.
                    NaN-filled zeros when coverage < 50%.

    Algorithm walkthrough:
        0. Crop images to remove frame number and focus on central region.
        1. Normalise frames to [0,1] in the cropped region.
        2. Compute image signal: 1 - NCC between consecutive frames.
        3. Detect heart rate via FFT spectral peak in the configured range.
        4. Bandpass filter both signals at [0.7, 2.2] x f_heart.

    Parameters
    ----------
    main_window : MainWindow
        Used to cache the gating signal in runtime_data.gating_signal.
    frames : (N, H, W) array
        IVUS pullback frames (uint8 or float32).
    report_data : pd.DataFrame
        Report data with columns ['frame', 'lumen_area'].
    lower_limit : int
        Frame number offset for gating signal (0-based).
    x1, x2, y1, y2 : int
        Crop coordinates for the image signal (default: 50:450, 400x400).
        *Reasoning:* to remove embedded frame number, and more signal in center,
        of the image.

    Returns
    -------
    image_raw : (N,) array
        Raw image signal (1 - NCC).
    contour_raw : (N,) array
        Raw contour signal (lumen area).
    image_filtered : (N,) array
        Bandpass-filtered image signal.
    contour_filtered : (N,) array
        Bandpass-filtered contour signal.
    """
    cfg = main_window.config.gating
    fs = main_window.runtime_data.metadata['frame_rate']
    N = len(frames)

    # ── Cache ──────────────────────────────────────────────────────────────
    try:
        gs = main_window.runtime_data.gating_signal
        if gs and gs.get('gating_config') == dict(vars(cfg)) and len(gs.get('image_based_gating', [])) == N:
            # Frequency sweep is recomputed even on cache hit: its BPM range
            # is fixed by compute_frequency_sweep(), not by gating_config, so
            # a project file saved under an older sweep range (e.g. before
            # the fixed 40-400 BPM window) would otherwise stay stale forever.
            image_raw_cached = np.array(gs['image_based_gating'])
            bpm_cuts, sweep = compute_frequency_sweep(image_raw_cached, fs)
            gs['freq_sweep_bpm_cuts'] = bpm_cuts.tolist()
            gs['freq_sweep_signals'] = sweep.tolist()
            return (
                image_raw_cached,
                np.array(gs['contour_based_gating']),
                np.array(gs['image_based_gating_filtered']),
                np.array(gs['contour_based_gating_filtered']),
            )
    except Exception:
        pass

    # ── Pre-process frames: crop -> [0,1] ──────────────────────────────────
    frames_crop = frames[:, x1:x2, y1:y2].astype(np.float32)
    f_lo = frames_crop.min(axis=(1, 2), keepdims=True)
    f_hi = frames_crop.max(axis=(1, 2), keepdims=True)
    f_rng = np.where(f_hi > f_lo, f_hi - f_lo, 1.0)
    frames_norm = (frames_crop - f_lo) / f_rng

    # ── Image signal: 1 - NCC ─────────────────────────────────────────────
    logger.info("Computing cross-correlation image signal …")
    image_raw = compute_correlation_signal(frames_norm)
    image_raw = normalize_data(image_raw, step=cfg.normalize_step)

    # ── Heart rate detection via FFT ───────────────────────────────────────
    f_heart_min = getattr(cfg, 'f_cardiac_min', 1.0)
    f_heart_max = getattr(cfg, 'f_cardiac_max', 3.5)
    f_heart = detect_heart_rate(image_raw, fs, f_heart_min, f_heart_max)

    # ── Bandpass filter ────────────────────────────────────────────────────
    lo_frac = getattr(cfg, 'bandpass_lo_frac', 0.7)
    hi_frac = getattr(cfg, 'bandpass_hi_frac', 2.2)

    image_filtered = bandpass_filter(image_raw, f_heart, fs, lo_frac, hi_frac)
    image_filtered = normalize_data(image_filtered, step=cfg.normalize_step)

    # ── Contour signal: lumen area ─────────────────────────────────────────
    contour_raw = np.zeros(N)
    contour_filtered = np.zeros(N)

    if report_data is not None and len(report_data) > 0:
        area_signal = compute_lumen_signal(report_data, N, lower_limit, key='lumen_area')
        minor_axis_signal = compute_lumen_signal(report_data, N, lower_limit, key='shortest_distance')
        has_contour = not np.all(np.isnan(area_signal)) and not np.all(np.isnan(minor_axis_signal))

        if has_contour:
            contour_raw = normalize_data(area_signal, step=cfg.normalize_step)
            contour_adjuster = normalize_data(minor_axis_signal, step=cfg.normalize_step)
            contour_raw = contour_raw * contour_adjuster  # adjust for only elliptic deformation, with same lumen area
            area_filtered = bandpass_filter(area_signal, f_heart, fs, lo_frac, hi_frac)
            contour_filtered = normalize_data(area_filtered, step=cfg.normalize_step)

    # ── Frequency sweep for interactive visualisation ──────────────────────
    bpm_cuts, sweep = compute_frequency_sweep(image_raw, fs)

    # ── Cache ─────────────────────────────────────────────────────────────
    main_window.runtime_data.gating_signal = {
        'image_based_gating': image_raw.tolist(),
        'contour_based_gating': contour_raw.tolist(),
        'image_based_gating_filtered': image_filtered.tolist(),
        'contour_based_gating_filtered': contour_filtered.tolist(),
        'gating_config': dict(vars(cfg)),
        'f_heart': f_heart,
        'f_heart_bpm': f_heart * 60,
        'freq_sweep_bpm_cuts': bpm_cuts.tolist(),
        'freq_sweep_signals': sweep.tolist(),
    }

    return image_raw, contour_raw, image_filtered, contour_filtered


def normalize_data(data, step: int) -> np.ndarray:
    """Z-score normalisation in non-overlapping windows of *step* frames."""
    data = np.asarray(data, dtype=float)
    if step == 0:
        std = np.nanstd(data)
        return (data - np.nanmean(data)) / std if std != 0 else np.zeros_like(data)
    out = np.zeros(len(data), dtype=float)
    for i in range(0, len(data), step):
        seg = data[i : i + step]
        std = np.nanstd(seg)
        if std != 0:
            out[i : i + step] = (seg - np.nanmean(seg)) / std
    return out


# ─────────────────────────────── image signal ────


def compute_correlation_signal(frames: np.ndarray) -> np.ndarray:
    """1 - normalised cross-correlation between consecutive frames (CCB s0 in Talou et al. 2015).

    High when frames differ (motion); low at stable end-systole/end-diastole.
    Frames should be [0,1]-normalised beforehand.
    """
    N = len(frames)
    s = np.zeros(N)
    for n in range(N - 1):
        u = frames[n].ravel().astype(float)
        v = frames[n + 1].ravel().astype(float)
        sig_u, sig_v = u.std(), v.std()
        if sig_u > 0 and sig_v > 0:
            corr = np.dot(u - u.mean(), v - v.mean()) / (sig_u * sig_v * len(u))
            s[n] = 1.0 - float(np.clip(corr, -1.0, 1.0))
    s[-1] = s[-2] if N > 1 else 0.0
    return s


def fft_peak_freq(
    signal: np.ndarray,
    fs: float,
    f_min: float,
    f_max: float,
    label: str = "spectral",
) -> float:
    """Dominant FFT spectral peak frequency within [f_min, f_max] Hz.

    Falls back to the band midpoint (with a warning) if the band contains no
    frequency bin, e.g. a signal too short for the given fs.
    """
    sig = np.nan_to_num(signal - np.nanmean(signal))
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(sig))
    mask = (freqs >= f_min) & (freqs <= f_max)
    if not mask.any():
        logger.warning(f"No {label} spectral peak in [{f_min}, {f_max}] Hz -> using midpoint")
        return (f_min + f_max) / 2
    return float(freqs[mask][np.argmax(spectrum[mask])])


def detect_heart_rate(
    signal: np.ndarray,
    fs: float,
    f_min: float = 1.0,
    f_max: float = 3.5,
) -> float:
    """Dominant heart rate via FFT spectral peak in physiological range [Hz]."""
    f_heart = fft_peak_freq(signal, fs, f_min, f_max, label="heart rate")
    logger.info(f"Heart rate: {f_heart:.3f} Hz  ({f_heart * 60:.0f} BPM)")

    # The 1-NCC signal has strong power at 2xf_heart (two motion events per cycle).
    # If the actual heart rate is below f_min/2, the harmonic rather than the
    # fundamental may be the dominant peak in [f_min, f_max].
    if f_heart > f_max / 2:
        logger.warning(
            f"Detected f_heart={f_heart:.2f} Hz is above half of f_max={f_max} Hz. "
            "The 2xf_heart harmonic of a slower heart rate may have been picked up. "
            "If gating looks wrong, try halving f_cardiac_max in config."
        )

    return f_heart


def bandpass_filter(
    signal: np.ndarray,
    f_heart: float,
    fs: float,
    lo_frac: float = 0.7,
    hi_frac: float = 2.2,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass at [lo_frac, hi_frac] x f_heart.

    Default passband [0.7, 2.2] x f_heart:
    - Lower cut removes slow pullback trend (DC and sub-cardiac drift).
    - Upper cut removes high-frequency speckle noise.
    - Passes up to 2nd harmonic so both sys and dia phases are visible
      as two distinct features per cardiac cycle in the image signal.
    """
    nyq = fs / 2.0
    lo = max(0.01, lo_frac * f_heart / nyq)
    hi = min(0.99, hi_frac * f_heart / nyq)
    if lo >= hi:
        logger.warning(f"Bandpass bounds invalid ({lo:.3f}, {hi:.3f}) -> returning raw signal")
        return np.array(signal, dtype=float)
    b, a = butter(order, [lo, hi], btype='band')
    try:
        return filtfilt(b, a, np.nan_to_num(signal))
    except Exception as exc:
        logger.warning(f"Bandpass filter failed: {exc} -> returning raw signal")
        return np.array(signal, dtype=float)


# ────────────────────────── contour: lumen area signal ────


def compute_lumen_signal(
    report_data,
    N: int,
    lower_limit: int,
    key: str = 'lumen_area',
) -> np.ndarray:
    """Extract e.g., lumen_area [mm²] from report_data into a length-N array.

    Frames without contours -> NaN; gaps are linearly interpolated so the
    bandpass filter operates on a complete signal.

    Returns NaN-filled array (all NaN) when coverage < 50%.
    """
    meas = np.full(N, np.nan)

    for _, row in report_data.iterrows():
        idx = int(row['frame']) - 1 - lower_limit
        if 0 <= idx < N:
            v = row.get(key, np.nan)
            if not pd.isna(v):
                meas[idx] = float(v)

    coverage = float(np.sum(~np.isnan(meas))) / N
    if coverage < 0.5:
        logger.info(f"{key.replace('_', ' ')} coverage {coverage:.0%} < 50% - contour signal omitted")
        return np.full(N, np.nan)

    logger.info(f"{key.replace('_', ' ')} coverage: {coverage:.0%}")

    # Linear interpolation over NaN gaps
    valid = np.where(~np.isnan(meas))[0]
    if len(valid) < 2:
        return np.full(N, np.nan)
    meas = np.interp(np.arange(N), valid, meas[valid])
    return meas


# ─────────────────────── frequency sweep for visualisation ────


def lowpass_filter(signal: np.ndarray, f_cutoff: float, fs: float, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass (used only for frequency sweep display)."""
    nyq = fs / 2.0
    Wn = min(f_cutoff / nyq, 0.99)
    b, a = butter(order, Wn, btype='low')
    try:
        return filtfilt(b, a, np.nan_to_num(signal))
    except Exception:
        return np.array(signal, dtype=float)


def filter_by_period(
    indices: np.ndarray,
    expected_interval: float,
    tolerance: float = 0.4,
) -> np.ndarray:
    """Drop peaks whose inter-peak gap violates the expected cardiac interval.

    Peaks closer than (1 - tolerance) x expected_interval to the previous
    kept peak are discarded as noise ripple or walk-algorithm duplicates.
    Peaks further than (1 + tolerance) x expected_interval are retained but
    logged as potential missed beats (no automatic gap-filling).

    Parameters
    ----------
    expected_interval : expected frames between consecutive peaks of this type
                        (e.g. fs / (2 x f_heart) for image-signal maxima).
    tolerance         : fractional tolerance; 0.4 -> ±40 %.
    """
    if len(indices) < 2:
        return indices

    t_min = (1.0 - tolerance) * expected_interval
    t_max = (1.0 + tolerance) * expected_interval
    kept = [int(indices[0])]
    for idx in indices[1:]:
        gap = int(idx) - kept[-1]
        if gap < t_min:
            logger.debug(f"Period filter: dropped peak at {idx} (gap {gap:.1f} < min {t_min:.1f})")
        else:
            if gap > t_max:
                logger.debug(
                    f"Period filter: large gap {gap:.1f} > {t_max:.1f} before peak at {idx} - possible missed beat"
                )
            kept.append(int(idx))
    return np.array(kept, dtype=int)


def compute_frequency_sweep(
    signal: np.ndarray,
    fs: float,
    n_steps: int = 30,
    bpm_lo: float = 40.0,
    bpm_hi: float = 400.0,
) -> tuple:
    """Low-pass filtered signal at n_steps cutoff frequencies (BPM labelled).

    Sweeps the fixed range (bpm_lo to bpm_hi) so the user can see the
    transition from over-smoothed (single hump/cycle) to noisy signal,
    regardless of the current heart/breathing rate estimate. Used both for
    the cardiac sweep (default 40-400 BPM; upper bound is 2 x the maximum
    physiological heart rate since the image signal has two peaks per
    cardiac cycle) and the breathing sweep (bpm_lo=10, bpm_hi=60 BrPM).
    Returns (bpm_cuts, sweep) where sweep[i] is z-scored signal at bpm_cuts[i].
    """
    f_lo = bpm_lo / 60.0
    f_hi = min(fs / 2 * 0.9, bpm_hi / 60.0)
    f_cuts = np.linspace(f_lo, f_hi, n_steps)
    bpm_cuts = f_cuts * 60  # BPM for display

    N = len(signal)
    sweep = np.zeros((n_steps, N))
    for i, f_c in enumerate(f_cuts):
        try:
            filt = lowpass_filter(signal, f_c, fs)
            std = filt.std()
            sweep[i] = (filt - filt.mean()) / std if std > 0 else filt
        except Exception:
            sweep[i] = 0.0
    return bpm_cuts, sweep
