"""Tests for the retained respiratory-signal helpers in gating.gating_pipeline.

The breathing-sort/en-bloc reordering was removed; these cover what remains and
feeds the (future) sorting: the detrend + respiratory extraction and the
peak/valley phase computation with manual hard-anchor override.
"""
import numpy as np

from gating.breathing_pipeline import (
    compute_breathing_signal,
    compute_breathing_phases,
    assign_breathing_bins,
    register_phase,
)


def test_breathing_signal_shapes_and_trend():
    frames = np.arange(300, dtype=float)
    areas = 3.0 + 0.01 * frames + 0.5 * np.sin(2 * np.pi * frames / 120.0)
    out = compute_breathing_signal(frames, areas, gated_frames=set(range(0, 300, 30)), fs=30.0)
    for key in ("frames", "areas", "trend", "slope", "residual", "smoothed", "f_resp"):
        assert key in out
    assert out["trend"].shape == frames.shape
    assert out["slope"].shape == frames.shape
    assert abs(np.median(out["slope"]) - 0.01) < 5e-3


def test_breathing_signal_degenerate():
    out = compute_breathing_signal(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert out["f_resp"] == 0.0
    assert np.allclose(out["residual"], 0.0)


def test_manual_anchors_override_auto():
    t = np.arange(600)
    sig = np.sin(2 * np.pi * t / 150.0)
    phase, peaks, valleys = compute_breathing_phases(
        sig, manual_peaks=[40], manual_valleys=[75], frames_arr=t, anchor_gap=20
    )
    assert 40 in peaks and 75 in valleys
    assert not [p for p in peaks if p != 40 and abs(p - 40) < 20]
    assert not [v for v in valleys if v != 75 and abs(v - 75) < 20]
    assert np.all((phase >= 0.0) & (phase < 1.0))


def test_manual_only_uses_labels_exclusively():
    t = np.arange(600)
    sig = np.sin(2 * np.pi * t / 150.0)  # would auto-detect several extrema
    peaks_frames = [50, 200, 350]
    valleys_frames = [125, 275, 425]
    phase, peaks, valleys = compute_breathing_phases(
        sig, manual_peaks=peaks_frames, manual_valleys=valleys_frames, frames_arr=t, manual_only=True
    )
    # only the labelled anchors survive
    assert sorted(peaks.tolist()) == peaks_frames
    assert sorted(valleys.tolist()) == valleys_frames
    assert np.all((phase >= 0.0) & (phase < 1.0))


def test_phases_without_anchors_are_valid():
    t = np.arange(600)
    sig = np.sin(2 * np.pi * t / 150.0)
    phase, peaks, valleys = compute_breathing_phases(sig, frames_arr=t)
    assert len(peaks) > 0 and len(valleys) > 0
    assert np.all((phase >= 0.0) & (phase < 1.0))


# ─────────────────────────── bin-registration sort ─────────────────────────


def _synthetic_gated_phase(seed=0):
    """Gated frames whose imaged area carries a one-directional breathing offset."""
    rng = np.random.default_rng(seed)
    N = 3000
    A_true = lambda x: 12.0 - 8.0 * (np.clip(x, 0, N) / N)  # noqa: E731  monotonic taper

    valleys = [0]
    while valleys[-1] < N:
        valleys.append(valleys[-1] + int(rng.integers(150, 230)))
    valleys = [v for v in valleys if v < N]
    peaks = [(valleys[i] + valleys[i + 1]) // 2 for i in range(len(valleys) - 1)]
    D = 200.0

    def disp(frames):
        frames = np.asarray(frames, float)
        out = np.zeros_like(frames)
        for i in range(len(valleys) - 1):
            v0, v1, p = valleys[i], valleys[i + 1], peaks[i]
            asc = (frames >= v0) & (frames < p)
            des = (frames >= p) & (frames < v1)
            out[asc] = D * (frames[asc] - v0) / max(1, (p - v0))
            out[des] = D * (v1 - frames[des]) / max(1, (v1 - p))
        return out

    dia = np.array([f for f in range(20, N - 20, 26)])
    area = A_true(dia + disp(dia)) + rng.normal(0, 0.05, len(dia))
    true_pos = dia + disp(dia)
    return dia, area, peaks, valleys, true_pos, N


def _inv(seq):
    seq = np.asarray(seq, float)
    return sum(1 for i in range(len(seq)) for j in range(i + 1, len(seq)) if seq[i] > seq[j])


def test_assign_breathing_bins_ranges():
    valleys = [0, 200]
    peaks = [100]
    frames = np.array([0, 25, 50, 75, 100, 125, 150, 175])
    bins = assign_breathing_bins(frames, peaks, valleys, n_bins=4)
    # valley (0) and just-before-next-valley (175) are bin 0; peak (100) is bin 3
    assert bins[0] == 0
    assert bins[frames.tolist().index(100)] == 3
    assert set(bins.tolist()) <= {0, 1, 2, 3}


def test_register_phase_recovers_order():
    dia, area, peaks, valleys, true_pos, N = _synthetic_gated_phase()
    R = register_phase(dia, area, peaks, valleys, n_bins=4, n_total=N)
    # per-bin shifts are monotonic and increase away from rest
    assert R['shifts'][0] == 0
    assert np.all(np.diff(np.abs(R['shifts'])) >= -1e-9)
    # sorting strongly reduces disorder of the true anatomical positions
    inv_raw = _inv(true_pos)
    inv_sorted = _inv(true_pos[R['order']])
    assert inv_sorted < 0.5 * inv_raw
    # resulting area sequence is near-monotonic
    mono = abs(np.corrcoef(area[R['order']], np.arange(len(area)))[0, 1])
    assert mono > 0.98


def test_register_phase_degenerate():
    R = register_phase(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), [], [], n_bins=4)
    assert list(R['order']) == [0, 1, 2]
