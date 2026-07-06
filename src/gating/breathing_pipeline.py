import numpy as np
from loguru import logger
from gating.automatic_gating import walk_extrema
from gating.gating_pipeline import lowpass_filter, fft_peak_freq


def compute_breathing_signal(
    frames_arr: np.ndarray,
    areas_arr: np.ndarray,
    gated_frames: set | None = None,
    fs: float = 30.0,
    f_heart: float | None = None,
    f_resp_override: float | None = None,
    poly_deg: int = 2,
    gated_weight: float = 10.0,
    cache: dict | None = None,
) -> dict:
    """
    Detrend the lumen-area sequence and extract the respiratory oscillation.

    Steps
    -----
    1. Fit a low-order polynomial trend (the pullback taper) to area vs. frame,
       giving *gated* (manually reviewed) frames ``gated_weight`` times the weight so
       the trend is anchored by reliable points and not dragged by ostial noise.
    2. Residual = area - trend.
    3. Detect the respiratory rate on the residual (or use ``f_resp_override``)
       and low-pass filter the residual at 2x that rate to isolate breathing.

    Parameters
    ----------
    frames_arr : (N,) array of frame numbers
    areas_arr : (N,) array of lumen areas (mm²)
    gated_frames : set of frame numbers to be given extra weight
    fs : sampling frequency (frames/sec)
    f_heart : heart rate (Hz) to expand the search range for breathing rate
    f_resp_override : override the detected breathing rate (Hz)
    poly_deg : degree of polynomial to fit the trend
    gated_weight : weight for gated frames
    cache : optional dict (typically ``runtime_data.gating_signal``) to read/write
        a cached result from. Callers like plot redraws invoke this on nearly
        every user action, so a signature over the inputs lets an unchanged
        pullback skip the polyfit + FFT + filtfilt work entirely.

    Returns
    -------
    dict with keys:
        frames : (N,) array of frame numbers
        areas : (N,) array of lumen areas (mm²)
        trend : (N,) polynomial trend fit to area vs. frame
        slope : (N,) derivative of the trend (mm²/frame)
        residual : (N,) area - trend
        smoothed : (N,) low-pass filtered residual at 2x breathing rate
        f_resp : detected or overridden breathing rate (Hz)
    """
    frames_arr = np.asarray(frames_arr, dtype=float)
    areas_arr = np.asarray(areas_arr, dtype=float)
    n_samples = len(frames_arr)

    signature = _build_signal_signature(
        frames_arr, areas_arr, gated_frames, fs, f_heart, f_resp_override, poly_deg, gated_weight
    )
    cached_result = _read_cached_signal(cache, signature)
    if cached_result is not None:
        return cached_result

    if n_samples < 3:
        result = _degenerate_signal_result(frames_arr, areas_arr)
    else:
        trend, slope, residual = _fit_weighted_trend(frames_arr, areas_arr, gated_frames, poly_deg, gated_weight)

        if f_resp_override is not None:
            f_resp = float(f_resp_override)
        else:
            f_resp = _detect_breathing_rate(residual, fs, f_heart)
        smoothed = _extract_breathing_signal(residual, f_resp, fs)

        result = {
            'frames': frames_arr,
            'areas': areas_arr,
            'trend': trend,
            'slope': slope,
            'residual': residual,
            'smoothed': smoothed,
            'f_resp': f_resp,
        }

    if cache is not None:
        cache['breathing_cache_signature'] = signature
        cache['breathing_cache_result'] = result

    return result


def _build_signal_signature(
    frames_arr: np.ndarray,
    areas_arr: np.ndarray,
    gated_frames: set | None,
    fs: float,
    f_heart: float | None,
    f_resp_override: float | None,
    poly_deg: int,
    gated_weight: float,
) -> dict:
    """Inputs that fully determine ``compute_breathing_signal``'s output, used to invalidate the cache."""
    return {
        'frames': frames_arr.tolist(),
        'areas': areas_arr.tolist(),
        'gated_frames': sorted(int(f) for f in gated_frames) if gated_frames else [],
        'fs': fs,
        'f_heart': f_heart,
        'f_resp_override': f_resp_override,
        'poly_deg': poly_deg,
        'gated_weight': gated_weight,
    }


def _read_cached_signal(cache: dict | None, signature: dict) -> dict | None:
    """Return the cached result if `signature` still matches, else None."""
    if cache is None or cache.get('breathing_cache_signature') != signature:
        return None
    cached_result = cache.get('breathing_cache_result')
    if cached_result is None:
        return None
    return {k: np.asarray(v) if isinstance(v, (list, np.ndarray)) else v for k, v in cached_result.items()}


def _degenerate_signal_result(frames_arr: np.ndarray, areas_arr: np.ndarray) -> dict:
    """Fallback result when there are too few points (<3) to fit a trend."""
    zeros = np.zeros(len(frames_arr))
    return {
        'frames': frames_arr,
        'areas': areas_arr,
        'trend': areas_arr.copy(),
        'slope': zeros,
        'residual': zeros,
        'smoothed': zeros,
        'f_resp': 0.0,
    }


def _fit_weighted_trend(
    frames_arr: np.ndarray,
    areas_arr: np.ndarray,
    gated_frames: set | None,
    poly_deg: int,
    gated_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Polynomial pullback-taper trend, its slope, and the area residual.

    Gated (manually reviewed) frames get ``gated_weight`` times the weight so
    the trend is anchored by reliable points and not dragged by ostial noise.
    """
    n_samples = len(frames_arr)
    weights = np.ones(n_samples)
    if gated_frames:
        weights = np.where(np.isin(frames_arr.astype(int), list(gated_frames)), gated_weight, 1.0)

    deg = min(poly_deg, n_samples - 1)
    coeffs = np.polyfit(frames_arr, areas_arr, deg=deg, w=weights)
    trend = np.polyval(coeffs, frames_arr)
    # Analytic derivative of the polynomial trend -> local taper slope (mm²/frame).
    slope = np.polyval(np.polyder(coeffs), frames_arr)
    residual = areas_arr - trend
    return trend, slope, residual


def _detect_breathing_rate(
    signal: np.ndarray,
    fs: float,
    f_heart_hz: float | None = None,
) -> float:
    """Dominant respiratory rate via FFT spectral peak [Hz].

    Searches 12-30 BrPM (0.20-0.50 Hz) by default.
    Expands upper bound to 1.0 Hz (60 BrPM) when heart rate > 100 BPM.
    """
    f_resp_min = 12 / 60  # 0.20 Hz
    f_resp_max = 30 / 60  # 0.50 Hz
    if f_heart_hz is not None and f_heart_hz * 60 > 100:
        f_resp_max = 60 / 60  # 1.00 Hz

    f_resp = fft_peak_freq(signal, fs, f_resp_min, f_resp_max, label="respiratory")
    logger.info(f"Respiratory rate: {f_resp:.3f} Hz  ({f_resp * 60:.1f} BrPM)")
    return f_resp


def _extract_breathing_signal(
    area_signal: np.ndarray,
    f_resp_hz: float,
    fs: float,
) -> np.ndarray:
    """Low-pass filter area signal to isolate the respiratory component.

    Cuts off at 2x the respiratory rate to retain the breathing oscillation
    while suppressing cardiac motion (~1 Hz and above).
    """
    f_cutoff = min(2.0 * f_resp_hz, fs / 2.0 * 0.9)
    return lowpass_filter(area_signal, f_cutoff, fs)


def compute_breathing_phases(
    breathing_signal: np.ndarray,
    manual_peaks: list[int] | None = None,
    manual_valleys: list[int] | None = None,
    frames_arr: np.ndarray | None = None,
    swing_fraction: float = 0.10,
    anchor_gap: int = 15,
    manual_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-frame breathing phase in [0, 1), peaks, and valley indices.

    Phase anchors: valleys (end-of-breath, vessel at rest) -> 0.0 / 1.0,
    peaks (mid-breath, vessel displaced) -> 0.5.  Phase between anchors is
    linearly interpolated from a monotonically increasing cumulative-phase
    sequence and then wrapped to [0, 1).

    Manual anchors override the automatic detection (hard anchors): any
    auto-detected extremum within ``anchor_gap`` samples of a manual anchor is
    discarded, so the automatic detector only *fills the gaps* between the
    frames the user labelled.  This is important in noisy ostial regions where
    the automatic detector is unreliable.

    Parameters
    ----------
    manual_peaks/manual_valleys : peak / valley locations **as frame numbers**
        (matched against ``frames_arr``).  If ``frames_arr`` is None they are
        treated as direct indices into ``breathing_signal``.

    Returns
    -------
    phase       : (N,) float array in [0, 1)
    peaks_idx   : indices of breathing peaks (auto + manual, vessel displaced)
    valleys_idx : indices of breathing valleys (auto + manual, end-of-breath)
    """
    sig_centered = np.nan_to_num(breathing_signal - np.nanmean(breathing_signal))
    n_samples = len(sig_centered)
    phase = np.zeros(n_samples)

    _, auto_peaks, auto_valleys = walk_extrema(sig_centered, swing_fraction=swing_fraction)
    manual_peak_idx = _frames_to_indices(manual_peaks, frames_arr, n_samples)
    manual_valley_idx = _frames_to_indices(manual_valleys, frames_arr, n_samples)

    peaks_idx, valleys_idx = _reconcile_extrema(
        auto_peaks, auto_valleys, manual_peak_idx, manual_valley_idx, manual_only, anchor_gap
    )

    anchors = _alternate_anchors(_sorted_anchor_points(peaks_idx, valleys_idx))
    if len(anchors) < 2:
        return phase, peaks_idx, valleys_idx

    anchor_frames, anchor_cum_phase = _cumulative_phase_anchors(anchors)
    phase_unwrapped = np.interp(
        np.arange(n_samples, dtype=float),
        anchor_frames,
        anchor_cum_phase,
        left=anchor_cum_phase[0],
        right=anchor_cum_phase[-1],
    )
    phase = phase_unwrapped % 1.0
    return phase, peaks_idx, valleys_idx


def _frames_to_indices(
    frame_numbers: list[int] | None,
    frames_arr: np.ndarray | None,
    n_samples: int,
) -> list[int]:
    """Map frame numbers to indices into the breathing signal (nearest match)."""
    if not frame_numbers:
        return []
    if frames_arr is None:
        return [int(f) for f in frame_numbers if 0 <= int(f) < n_samples]
    frames_arr = np.asarray(frames_arr)
    return [int(np.argmin(np.abs(frames_arr - f))) for f in frame_numbers]


def _reconcile_extrema(
    auto_peaks: np.ndarray,
    auto_valleys: np.ndarray,
    manual_peak_idx: list[int],
    manual_valley_idx: list[int],
    manual_only: bool,
    anchor_gap: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine auto-detected and manual extrema, letting manual anchors win.

    ``manual_only=True``: use only the user's labels.  Otherwise, auto extrema
    within ``anchor_gap`` samples of a manual anchor are dropped so manual
    anchors override the automatic detector locally, which only fills the
    gaps elsewhere.
    """
    if manual_only:
        peaks_idx = np.array(sorted(set(manual_peak_idx)), dtype=int)
        valleys_idx = np.array(sorted(set(manual_valley_idx)), dtype=int)
        return peaks_idx, valleys_idx

    manual_all = manual_peak_idx + manual_valley_idx

    def _drop_near_manual(auto_list):
        return [int(a) for a in auto_list if not manual_all or min(abs(int(a) - m) for m in manual_all) >= anchor_gap]

    peaks_idx = np.array(sorted(set(_drop_near_manual(auto_peaks) + manual_peak_idx)), dtype=int)
    valleys_idx = np.array(sorted(set(_drop_near_manual(auto_valleys) + manual_valley_idx)), dtype=int)
    return peaks_idx, valleys_idx


def _sorted_anchor_points(peaks_idx: np.ndarray, valleys_idx: np.ndarray) -> list[tuple[int, str]]:
    """Merge peak/valley indices into one (index, type) list sorted by index."""
    return sorted(
        [(int(f), 'v') for f in valleys_idx] + [(int(f), 'p') for f in peaks_idx],
        key=lambda point: point[0],
    )


def _cumulative_phase_anchors(anchors: list[tuple[int, str]]) -> tuple[np.ndarray, np.ndarray]:
    """Assign monotonically increasing cumulative phase to alternating anchors.

    Each peak<->valley step is a half breathing cycle (+0.5); the running
    total is wrapped to [0, 1) by the caller after interpolation.
    """
    anchor_frames: list[float] = []
    cum_phase: list[float] = []
    cum = 0.0 if anchors[0][1] == 'v' else 0.5
    for k, (frame, _) in enumerate(anchors):
        if k > 0:
            cum += 0.5
        anchor_frames.append(float(frame))
        cum_phase.append(cum)
    return np.array(anchor_frames), np.array(cum_phase)


def _alternate_anchors(points: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Force a peak/valley point list into a strictly alternating sequence.

    Consecutive same-type anchors get a synthetic opposite anchor inserted at
    their midpoint, so the half-cycle phase model (0.5 per step) stays valid
    even when the user labels two peaks with no valley between them.
    """
    if len(points) < 2:
        return points
    out: list[tuple[int, str]] = [points[0]]
    for frame, kind in points[1:]:
        prev_frame, prev_kind = out[-1]
        if kind == prev_kind and frame != prev_frame:
            mid = (prev_frame + frame) // 2
            if mid not in (prev_frame, frame):
                out.append((mid, 'p' if kind == 'v' else 'v'))
        out.append((frame, kind))
    return out


# ─────────────── breathing-bin registration sort (gated frames only) ────
#
# Idea (per phase, diastole and systole are handled independently):
#   * The labelled valleys are the vessel at rest (displacement 0); peaks are max
#     displacement. Each breathing half-cycle (valley->peak and peak->valley) is
#     split into the same number of bins, and by symmetry the ascending and
#     descending bin at a given displacement are pooled together.
#   * Bin 0 (the valleys) is the ground truth: fit lumen-area (divided by ) vs frame position
#     through those rest frames.
#   * Every other bin's frames sample the SAME vessel profile but shifted along
#     the pullback by the breathing displacement.  Slide that bin's area profile
#     until it best matches the ground-truth curve; the winning shift is the
#     displacement, which is subtracted to recover each frame's true position.
#   * Frames are then re-ordered by corrected position (gaps ignored).


def assign_breathing_bins(
    frames: np.ndarray,
    peaks: list[int],
    valleys: list[int],
    n_bins: int = 5,
) -> np.ndarray:
    """Assign each frame to a displacement bin 0..n_bins-1 (0 = valley / rest).
    Bins are needed to adjust for unregular breathing, were longer and shorter half-cycles
    can occur. The binning harmonizes the displacement.

    Ascending (valley->peak) maps displacement 0->max as bin 0->n_bins-1; descending
    (peak->valley) is mirrored so it shares the same bins by displacement.  Frames
    outside any labelled half-cycle get -1.
    """
    anchors = sorted([(int(v), 'v') for v in valleys] + [(int(p), 'p') for p in peaks])
    if len(anchors) < 2:
        return np.full(len(frames), -1, dtype=int)
    anchor_frames = np.array([frame for frame, _ in anchors])
    bins = np.full(len(frames), -1, dtype=int)
    for i, frame in enumerate(frames):
        anchor_idx = int(np.searchsorted(anchor_frames, frame, side='right')) - 1
        if anchor_idx < 0 or anchor_idx >= len(anchors) - 1:
            continue
        start_frame, start_kind = anchors[anchor_idx]
        end_frame, _ = anchors[anchor_idx + 1]
        if end_frame <= start_frame:
            continue
        progress = (frame - start_frame) / (end_frame - start_frame)  # position within the half-cycle, 0..1
        bin_idx = int(np.floor(progress * n_bins)) if start_kind == 'v' else int(np.floor((1.0 - progress) * n_bins))
        bins[i] = min(max(bin_idx, 0), n_bins - 1)
    return bins


def register_phase(
    frames: np.ndarray,
    areas: np.ndarray,
    peaks: list[int],
    valleys: list[int],
    n_bins: int = 4,
    gt_deg: int = 3,
    max_shift: float | None = None,
    n_total: int | None = None,
    enforce_monotonic: bool = True,
) -> dict:
    """Breathing-corrected positions for one gated phase via bin registration.

    See the module comment above.  Returns a dict with ``bins`` (per-frame bin),
    ``shifts`` (per-bin displacement in frames), ``corrected`` (per-frame
    corrected position) and ``order`` (frame indices sorted by corrected pos).
    """
    frames = np.asarray(frames, dtype=float)
    areas = np.asarray(areas, dtype=float)
    n_samples = len(frames)
    if n_samples < gt_deg + 2:
        return _passthrough_registration(frames, np.zeros(n_samples, int), max(n_bins, 1))

    bins = assign_breathing_bins(frames, peaks, valleys, n_bins)
    if max_shift is None:
        span = (n_total if n_total else int(frames.max())) or 1
        max_shift = span / 4.0

    ground_truth = _fit_ground_truth_curve(frames, areas, bins, gt_deg)
    if ground_truth is None:  # too few rest frames -> no reliable reference
        return _passthrough_registration(frames, bins, n_bins)
    ground_truth_area = ground_truth

    shifts = _estimate_bin_shifts(frames, areas, bins, n_bins, max_shift, ground_truth_area)
    if enforce_monotonic:
        shifts = _enforce_monotonic_shifts(shifts)

    corrected = _apply_bin_shifts(frames, bins, shifts, n_bins)
    order = np.argsort(corrected, kind='stable')
    return {'bins': bins, 'shifts': shifts, 'corrected': corrected, 'order': order}


def _passthrough_registration(frames: np.ndarray, bins: np.ndarray, n_shift_bins: int) -> dict:
    """No-correction fallback (identity mapping) for when there isn't enough data to register."""
    return {
        'bins': bins,
        'shifts': np.zeros(n_shift_bins),
        'corrected': frames.copy(),
        'order': np.argsort(frames, kind='stable'),
    }


def _fit_ground_truth_curve(frames: np.ndarray, areas: np.ndarray, bins: np.ndarray, gt_deg: int):
    """Fit the rest-frame (bin 0) area-vs-frame curve that every other bin is matched against.

    Widens to bin<=0 if bin 0 alone has too few points for the fit.  Returns
    a callable ``area(frame)`` (clipped to the fitted frame range), or None
    if there still aren't enough rest frames for a reliable fit.
    """
    is_rest = bins == 0
    if is_rest.sum() < gt_deg + 1:
        is_rest = bins <= 0
    if is_rest.sum() < gt_deg + 1:
        return None
    coeffs = np.polyfit(frames[is_rest], areas[is_rest], gt_deg)
    lo, hi = frames[is_rest].min(), frames[is_rest].max()

    def ground_truth_area(x):
        return np.polyval(coeffs, np.clip(x, lo, hi))

    return ground_truth_area


def _estimate_bin_shifts(
    frames: np.ndarray,
    areas: np.ndarray,
    bins: np.ndarray,
    n_bins: int,
    max_shift: float,
    ground_truth_area,
) -> np.ndarray:
    """Per-bin displacement that best aligns each bin's area profile with the rest-frame ground truth."""
    shift_grid = np.arange(-max_shift, max_shift + 1.0, 2.0)
    shifts = np.zeros(n_bins)
    for bin_idx in range(1, n_bins):
        in_bin = bins == bin_idx
        if in_bin.sum() < 3:
            shifts[bin_idx] = shifts[bin_idx - 1]  # too few frames -> reuse previous bin's shift
            continue
        bin_frames, bin_areas = frames[in_bin], areas[in_bin]
        errors = [np.mean((bin_areas - ground_truth_area(bin_frames + s)) ** 2) for s in shift_grid]
        shifts[bin_idx] = float(shift_grid[int(np.argmin(errors))])
    return shifts


def _enforce_monotonic_shifts(shifts: np.ndarray) -> np.ndarray:
    """Keep |shift| non-decreasing and of one consistent sign across bins.

    Displacement grows away from the valley (bin 0); the sign used is the
    dominant direction across the bins.
    """
    sign = 1.0 if np.sum(shifts) >= 0 else -1.0
    magnitudes = np.maximum.accumulate(np.abs(shifts))
    return sign * magnitudes


def _apply_bin_shifts(frames: np.ndarray, bins: np.ndarray, shifts: np.ndarray, n_bins: int) -> np.ndarray:
    """Shift each frame's position by its bin's displacement to recover the true pullback position."""
    corrected = frames.copy()
    for bin_idx in range(n_bins):
        corrected[bins == bin_idx] = frames[bins == bin_idx] + shifts[bin_idx]
    return corrected
