import numpy as np
from loguru import logger
from gating.automatic_gating import walk_extrema
from gating.gating_pipeline import lowpass_filter


def compute_breathing_signal(
    frames_arr: np.ndarray,
    areas_arr: np.ndarray,
    gated_frames: set | None = None,
    fs: float = 30.0,
    f_heart: float | None = None,
    f_resp_override: float | None = None,
    poly_deg: int = 2,
    gated_weight: float = 10.0,
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

    Returns a dict with keys: ``frames``, ``areas``, ``trend``, ``slope``
    (dArea/dFrame of the trend), ``residual``, ``smoothed`` (breathing wave),
    and ``f_resp`` (Hz).  The trend slope is what converts an area residual into
    a longitudinal position offset for the sort.
    """
    frames_arr = np.asarray(frames_arr, dtype=float)
    areas_arr = np.asarray(areas_arr, dtype=float)
    n = len(frames_arr)
    if n < 3:
        zeros = np.zeros(n)
        return {
            'frames': frames_arr,
            'areas': areas_arr,
            'trend': areas_arr.copy(),
            'slope': zeros,
            'residual': zeros,
            'smoothed': zeros,
            'f_resp': 0.0,
        }

    weights = np.ones(n)
    if gated_frames:
        weights = np.where(np.isin(frames_arr.astype(int), list(gated_frames)), gated_weight, 1.0)

    deg = min(poly_deg, n - 1)
    coeffs = np.polyfit(frames_arr, areas_arr, deg=deg, w=weights)
    trend = np.polyval(coeffs, frames_arr)
    # Analytic derivative of the polynomial trend -> local taper slope (mm²/frame).
    slope = np.polyval(np.polyder(coeffs), frames_arr)
    residual = areas_arr - trend

    if f_resp_override is not None:
        f_resp = float(f_resp_override)
    else:
        f_resp = detect_breathing_rate(residual, fs, f_heart)
    smoothed = extract_breathing_signal(residual, f_resp, fs)

    return {
        'frames': frames_arr,
        'areas': areas_arr,
        'trend': trend,
        'slope': slope,
        'residual': residual,
        'smoothed': smoothed,
        'f_resp': f_resp,
    }


def detect_breathing_rate(
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

    sig = np.nan_to_num(signal - np.nanmean(signal))
    freqs = np.fft.rfftfreq(len(sig), d=1.0 / fs)
    spectrum = np.abs(np.fft.rfft(sig))
    mask = (freqs >= f_resp_min) & (freqs <= f_resp_max)
    if not mask.any():
        logger.warning(f"No respiratory spectral peak in [{f_resp_min:.2f}, {f_resp_max:.2f}] Hz")
        return (f_resp_min + f_resp_max) / 2
    f_resp = float(freqs[mask][np.argmax(spectrum[mask])])
    logger.info(f"Respiratory rate: {f_resp:.3f} Hz  ({f_resp * 60:.1f} BrPM)")
    return f_resp


def extract_breathing_signal(
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


def _alternate_anchors(points: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Force a peak/valley point list into a strictly alternating sequence.

    Consecutive same-type anchors get a synthetic opposite anchor inserted at
    their midpoint, so the half-cycle phase model (0.5 per step) stays valid
    even when the user labels two peaks with no valley between them.
    """
    if len(points) < 2:
        return points
    out: list[tuple[int, str]] = [points[0]]
    for f, t in points[1:]:
        pf, pt = out[-1]
        if t == pt and f != pf:
            mid = (pf + f) // 2
            if mid not in (pf, f):
                out.append((mid, 'p' if t == 'v' else 'v'))
        out.append((f, t))
    return out


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
    sig = np.nan_to_num(breathing_signal - np.nanmean(breathing_signal))
    N = len(sig)
    phase = np.zeros(N)

    _, auto_peaks, auto_valleys = walk_extrema(sig, swing_fraction=swing_fraction)

    # Map manual frame numbers -> signal indices.
    def _to_idx(frame_numbers):
        if not frame_numbers:
            return []
        if frames_arr is None:
            return [int(f) for f in frame_numbers if 0 <= int(f) < N]
        fa = np.asarray(frames_arr)
        return [int(np.argmin(np.abs(fa - f))) for f in frame_numbers]

    man_peak_idx = _to_idx(manual_peaks)
    man_valley_idx = _to_idx(manual_valleys)
    manual_all = man_peak_idx + man_valley_idx

    if manual_only:
        # Use ONLY the user's labels — no automatic detection at all. Everything
        # shown is then a real, deletable label and nothing hidden drives phase.
        peaks_idx = np.array(sorted(set(man_peak_idx)), dtype=int)
        valleys_idx = np.array(sorted(set(man_valley_idx)), dtype=int)
    else:
        # Drop auto extrema that sit near any manual anchor (manual wins).
        def _keep(auto_list):
            return [
                int(a) for a in auto_list if not manual_all or min(abs(int(a) - m) for m in manual_all) >= anchor_gap
            ]

        peaks_idx = np.array(sorted(set(_keep(auto_peaks) + man_peak_idx)), dtype=int)
        valleys_idx = np.array(sorted(set(_keep(auto_valleys) + man_valley_idx)), dtype=int)

    all_pts = sorted(
        [(int(f), 'v') for f in valleys_idx] + [(int(f), 'p') for f in peaks_idx],
        key=lambda x: x[0],
    )
    all_pts = _alternate_anchors(all_pts)

    if len(all_pts) < 2:
        return phase, peaks_idx, valleys_idx

    anchors_frame: list[float] = []
    anchors_cum: list[float] = []
    cum = 0.0 if all_pts[0][1] == 'v' else 0.5
    for k, (f, _) in enumerate(all_pts):
        if k > 0:
            cum += 0.5
        anchors_frame.append(float(f))
        anchors_cum.append(cum)

    af = np.array(anchors_frame)
    ac = np.array(anchors_cum)
    phase_unwrapped = np.interp(np.arange(N, dtype=float), af, ac, left=ac[0], right=ac[-1])
    phase = phase_unwrapped % 1.0
    return phase, peaks_idx, valleys_idx


# ─────────────── breathing-bin registration sort (gated frames only) ────
#
# Idea (per phase — diastole and systole are handled independently):
#   * The labelled valleys are the vessel at rest (displacement 0); peaks are max
#     displacement.  Each breathing half-cycle (valley->peak and peak->valley) is
#     split into the same number of bins, and by symmetry the ascending and
#     descending bin at a given displacement are pooled together.
#   * Bin 0 (the valleys) is the ground truth: fit lumen-area vs frame position
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
    n_bins: int = 4,
) -> np.ndarray:
    """Assign each frame to a displacement bin 0..n_bins-1 (0 = valley / rest).

    Ascending (valley->peak) maps displacement 0->max as bin 0->n_bins-1; descending
    (peak->valley) is mirrored so it shares the same bins by displacement.  Frames
    outside any labelled half-cycle get -1.
    """
    anchors = sorted([(int(v), 'v') for v in valleys] + [(int(p), 'p') for p in peaks])
    if len(anchors) < 2:
        return np.full(len(frames), -1, dtype=int)
    af = np.array([a for a, _ in anchors])
    bins = np.full(len(frames), -1, dtype=int)
    for i, f in enumerate(frames):
        j = int(np.searchsorted(af, f, side='right')) - 1
        if j < 0 or j >= len(anchors) - 1:
            continue
        a0, t0 = anchors[j]
        a1, _ = anchors[j + 1]
        if a1 <= a0:
            continue
        u = (f - a0) / (a1 - a0)  # position within the half-cycle, 0..1
        b = int(np.floor(u * n_bins)) if t0 == 'v' else int(np.floor((1.0 - u) * n_bins))
        bins[i] = min(max(b, 0), n_bins - 1)
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
    n = len(frames)
    if n < gt_deg + 2:
        return {
            'bins': np.zeros(n, int),
            'shifts': np.zeros(max(n_bins, 1)),
            'corrected': frames.copy(),
            'order': np.argsort(frames, kind='stable'),
        }

    bins = assign_breathing_bins(frames, peaks, valleys, n_bins)
    if max_shift is None:
        span = (n_total if n_total else int(frames.max())) or 1
        max_shift = span / 4.0

    # Ground truth = rest (bin 0) frames; widen if too few for the fit.
    gt = bins == 0
    if gt.sum() < gt_deg + 1:
        gt = bins <= 0
    if gt.sum() < gt_deg + 1:  # still too few -> no reliable reference
        return {
            'bins': bins,
            'shifts': np.zeros(n_bins),
            'corrected': frames.copy(),
            'order': np.argsort(frames, kind='stable'),
        }
    cg = np.polyfit(frames[gt], areas[gt], gt_deg)
    gt_lo, gt_hi = frames[gt].min(), frames[gt].max()

    def A_gt(x):
        return np.polyval(cg, np.clip(x, gt_lo, gt_hi))

    grid = np.arange(-max_shift, max_shift + 1.0, 2.0)
    shifts = np.zeros(n_bins)
    for k in range(1, n_bins):
        m = bins == k
        if m.sum() < 3:
            shifts[k] = shifts[k - 1]  # too few frames -> reuse previous bin's shift
            continue
        fk, ak = frames[m], areas[m]
        errs = [np.mean((ak - A_gt(fk + s)) ** 2) for s in grid]
        shifts[k] = float(grid[int(np.argmin(errs))])

    if enforce_monotonic:
        # displacement grows away from the valley; keep |shift| non-decreasing and
        # of one consistent sign (the dominant direction across the bins)
        sign = 1.0 if np.sum(shifts) >= 0 else -1.0
        mags = np.maximum.accumulate(np.abs(shifts))
        shifts = sign * mags

    corrected = frames.copy()
    for k in range(n_bins):
        corrected[bins == k] = frames[bins == k] + shifts[k]
    order = np.argsort(corrected, kind='stable')
    return {'bins': bins, 'shifts': shifts, 'corrected': corrected, 'order': order}
