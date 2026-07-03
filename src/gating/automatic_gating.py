import numpy as np
from loguru import logger

from gating.gating_pipeline import filter_by_period


def walk_extrema(
    signal: np.ndarray,
    swing_fraction: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hysteresis-gated turning-point detector.

    Walks the signal and registers a local maximum only after the value has
    dropped more than *swing_fraction* x peak-to-peak below the running high
    since the last confirmed turning point; vice versa for minima.
    *Note: Tested performance against scipy.signal.find_peaks()
    and found it more robust for IVUS gating signals.*

    Advantages over scipy find_peaks:
    - No global height threshold: amplitude-agnostic, adapts to signal level.
    - No minimum-distance parameter: the hysteresis swing alone suppresses
    micro-wiggles caused by residual noise on the bandpass-filtered signal.
    - Produces a naturally alternating max / min / max / min sequence that
    maps directly to cardiac phases without post-hoc alternation heuristics.

    Parameters
    ----------
    signal         : 1-D array (bandpass-filtered, normalised).
    swing_fraction : fraction of peak-to-peak range a reversal must exceed
                    before a turning point is registered.  0.15 (15%) works
                    well for IVUS bandpass-filtered signals.

    Returns
    -------
    all_extrema_idx : sorted union of maxima and minima indices
    maxima_idx      : indices of local maxima (high motion / end-phase peaks)
    minima_idx      : indices of local minima
    """
    sig = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    ptp = float(sig.max() - sig.min())
    empty: np.ndarray = np.array([], dtype=int)
    if ptp == 0.0:
        return empty, empty, empty

    threshold = swing_fraction * ptp
    maxima: list[int] = []
    minima: list[int] = []

    # direction: +1 = currently tracking a rising run (looking for maximum)
    #            -1 = currently tracking a falling run (looking for minimum)
    #            None = not yet determined (waiting for first significant swing)
    direction: int | None = None
    extreme_val = sig[0]
    extreme_idx = 0

    for i in range(1, len(sig)):
        v = sig[i]
        if direction is None:
            if v - sig[0] >= threshold:
                direction = 1
                extreme_val, extreme_idx = v, i
            elif sig[0] - v >= threshold:
                direction = -1
                extreme_val, extreme_idx = v, i
        elif direction == 1:
            if v > extreme_val:
                extreme_val, extreme_idx = v, i
            elif extreme_val - v >= threshold:
                maxima.append(extreme_idx)
                direction = -1
                extreme_val, extreme_idx = v, i
        else:  # direction == -1
            if v < extreme_val:
                extreme_val, extreme_idx = v, i
            elif v - extreme_val >= threshold:
                minima.append(extreme_idx)
                direction = 1
                extreme_val, extreme_idx = v, i

    maxima_idx = np.array(maxima, dtype=int)
    minima_idx = np.array(minima, dtype=int)
    all_extrema = np.sort(np.concatenate([maxima_idx, minima_idx]))
    return all_extrema, maxima_idx, minima_idx


class AutomaticGating:
    def __init__(self, main_window, report_data, lower_limit: int = 0) -> None:
        self.main_window = main_window
        self.report_data = report_data
        self.lower_limit = lower_limit

    def _signal_to_frame_key(self, sig_idx: int) -> int:
        return self.lower_limit + int(sig_idx)

    def automatic_gating(self, image_filtered: np.ndarray, contour_filtered: np.ndarray) -> None:
        """Classify frames as systole or diastole.

        Combined path (contour available)
        ----------------------------------
        1. Walk the image signal with hysteresis -> image-signal minima (minimum-
           motion stable frames = end-diastole / end-systole, 2 per cardiac cycle).
        2. Walk the contour (area) signal independently -> area maxima (end-
           diastole, lumen largest) and area minima (end-systole, lumen smallest).
        3. Apply a period-consistency filter to each peak list using f_heart so
           that noise ripples and duplicate detections are removed.
        4. Classify each image valley by the nearest area extremum: area maximum ->
           diastole, area minimum -> systole.

        Image-only path (no contour)
        ----------------------------
        Walk the image signal; alternate valleys into two groups; classify by mean
        amplitude (lower = more stable = diastole).
        """
        gs = getattr(self.main_window.runtime_data, 'gating_signal', None) or {}
        f_heart = gs.get('f_heart')
        fs = float(self.main_window.runtime_data.metadata.get('frame_rate', 30))

        # Expected inter-peak/valley interval for image-signal maxima: 2 peaks per cycle
        T_half = fs / (2.0 * f_heart) if f_heart and f_heart > 0 else None

        # ── Walk image signal - track valleys (stable end-phases) ─────────
        _, _, img_minima = walk_extrema(image_filtered)
        if T_half is not None:
            img_minima = filter_by_period(img_minima, T_half)

        if len(img_minima) < 2:
            logger.warning('Auto-gating: too few image signal valleys - check signal quality')
            return

        # ── Classify ───────────────────────────────────────────────────────
        contour_flat = np.all(np.abs(contour_filtered) < 1e-9)

        if not contour_flat:
            T_full = T_half * 2 if T_half is not None else None
            _, area_maxima, area_minima = walk_extrema(contour_filtered)
            if T_full is not None:
                area_maxima = filter_by_period(area_maxima, T_full)
                area_minima = filter_by_period(area_minima, T_full)

            logger.info(
                f'Walk extrema: {len(img_minima)} image valleys, '
                f'{len(area_maxima)} area maxima, {len(area_minima)} area minima'
            )

            dia_frames, sys_frames = self._classify_by_area_extrema(img_minima, area_maxima, area_minima)
            if not dia_frames or not sys_frames:
                logger.warning('Area-extrema classification collapsed - falling back to alternating')
                dia_frames, sys_frames = self._alternate_by_amplitude(img_minima, image_filtered)
        else:
            logger.info(f'Walk extrema (image-only): {len(img_minima)} image valleys')
            dia_frames, sys_frames = self._alternate_by_amplitude(img_minima, image_filtered)

        if not dia_frames and not sys_frames:
            logger.warning('Auto-gating produced no frames - check signal quality')
            return

        self._apply_gating(dia_frames, sys_frames)

    # ── combined: image peaks + area extrema ──────────────────────────────

    def _classify_by_area_extrema(
        self,
        img_maxima: np.ndarray,
        area_maxima: np.ndarray,
        area_minima: np.ndarray,
    ) -> tuple[list, list]:
        """Classify each image-signal peak by the nearest area-signal extremum.

        Nearest area maximum -> lumen area near its peak -> end-systole.
        Nearest area minimum -> lumen area near its valley -> end-diastole.

        Uses global nearest-neighbour (no distance cutoff) so the method
        degrades gracefully when only a few area extrema are detected.
        """
        all_area = np.concatenate([area_maxima, area_minima])
        if len(all_area) == 0:
            return [], []

        # Aortic vessel dilates during systole → area maximum = systole (+1)
        # and contracts during diastole → area minimum = diastole (-1)
        tags = np.concatenate(
            [
                np.ones(len(area_maxima), dtype=int),  # area_maxima → systole
                -np.ones(len(area_minima), dtype=int),  # area_minima → diastole
            ]
        )
        order = np.argsort(all_area)
        all_area = all_area[order]
        tags = tags[order]

        dia_frames: list[int] = []
        sys_frames: list[int] = []
        for i in img_maxima:
            nearest = int(np.argmin(np.abs(all_area - i)))
            frame_key = self._signal_to_frame_key(i)
            if tags[nearest] == -1:  # area minimum = diastole
                dia_frames.append(frame_key)
            else:
                sys_frames.append(frame_key)

        logger.info(f'Area-extrema classification: {len(dia_frames)} dia, {len(sys_frames)} sys')
        return dia_frames, sys_frames

    # ── image-only: alternating peaks ─────────────────────────────────────

    def _alternate_by_amplitude(self, maxima: np.ndarray, image_filtered: np.ndarray) -> tuple[list, list]:
        """Alternate image-signal peaks; lower mean amplitude -> sharpest images -> diastole."""
        first_sig = maxima[::2].tolist()
        second_sig = maxima[1::2].tolist()

        first_frames = [self._signal_to_frame_key(i) for i in first_sig]
        second_frames = [self._signal_to_frame_key(i) for i in second_sig]

        N = len(image_filtered)
        amp_f = self._mean_amp(image_filtered, first_frames, N)
        amp_s = self._mean_amp(image_filtered, second_frames, N)

        if amp_f <= amp_s:
            dia_frames, sys_frames = first_frames, second_frames
        else:
            dia_frames, sys_frames = second_frames, first_frames

        logger.info(
            f'Image-only gating: {len(dia_frames)} dia, {len(sys_frames)} sys '
            f'(amp_dia={min(amp_f, amp_s):.3f}, amp_sys={max(amp_f, amp_s):.3f})'
        )
        return dia_frames, sys_frames

    def _mean_amp(self, signal: np.ndarray, frame_keys: list, N: int) -> float:
        idxs = [k - self.lower_limit for k in frame_keys if 0 <= k - self.lower_limit < N]
        return float(np.mean([signal[i] for i in idxs])) if idxs else 0.0

    def _apply_gating(self, dia_frames: list, sys_frames: list) -> None:
        for fd in self.main_window.runtime_data.frame_data_dct.values():
            fd.phase = '-'
        self.main_window.runtime_data.gated_frames_dia = []
        self.main_window.runtime_data.gated_frames_sys = []
        self.main_window.diastolic_frame_box.setChecked(False)
        self.main_window.systolic_frame_box.setChecked(False)

        self.main_window.runtime_data.gated_frames_dia = sorted(dia_frames)
        self.main_window.runtime_data.gated_frames_sys = sorted(sys_frames)

        for frame in self.main_window.runtime_data.gated_frames_dia:
            if frame in self.main_window.runtime_data.frame_data_dct:
                self.main_window.runtime_data.frame_data_dct[frame].phase = 'D'
        for frame in self.main_window.runtime_data.gated_frames_sys:
            if frame in self.main_window.runtime_data.frame_data_dct:
                self.main_window.runtime_data.frame_data_dct[frame].phase = 'S'

        logger.info(f'Gating applied: {len(dia_frames)} diastolic, {len(sys_frames)} systolic')
