from __future__ import annotations
from typing import Any, TypedDict

import numpy as np

from domain.io_types import FrameData


class CctaRuntimeData:
    def __init__(self):
        self.metadata: dict = {}
        self.volume: np.ndarray | None = None  # (Z, Y, X) int16 HU
        self.voxel_spacing: tuple[float, float, float] | None = None  # (dz, dy, dx) mm
        self.mask: np.ndarray | None = None  # (Z, Y, X) uint8 label values
        self.labels: list[int] = []  # non-background labels present in mask


class GatingSignal(TypedDict, total=False):
    image_based_gating: list[float]
    contour_based_gating: list[float]
    image_based_gating_filtered: list[float]
    contour_based_gating_filtered: list[float]
    gating_config: dict[str, Any]
    f_heart: float
    f_heart_bpm: float
    freq_sweep_bpm_cuts: list[float]
    freq_sweep_signals: list[float]
    f_resp: float
    f_resp_override: float
    breathing_cache_signature: dict[str, Any]
    breathing_cache_result: dict[str, Any]
    breathing_residual: list[float]
    breathing_frames: list[int]
    breathing_display_signal: list[float]
    breathing_phase: list[float]
    breathing_auto_peaks: list[int]
    breathing_auto_valleys: list[int]
    breathing_manual_mode: bool
    breathing_manual_peaks: list[int]
    breathing_manual_valleys: list[int]
    sort_signature: dict[str, Any]
    sort_peaks: list[int]
    sort_valleys: list[int]
    sort_n_bins: int
    sort_dia_order: list[int]
    sort_sys_order: list[int]
    sort_dia_pos: list[list[float]]
    sort_sys_pos: list[list[float]]
    sort_dia_shifts: list[float]
    sort_sys_shifts: list[float]


class RuntimeData:
    def __init__(self):
        self.frame_data_dct: dict[int, FrameData] | None = None
        self.metadata: dict = {}
        self.images: np.ndarray | None = None
        self.images_rgb: np.ndarray | None = None
        self.gated_frames: list[int] = []
        self.gated_frames_dia: list[int] = []
        self.gated_frames_sys: list[int] = []
        self.tagged_frames: list[int] = []
        self.tmp_contours: dict[str, tuple[list[float], list[float]]] = {}
        self.gating_signal: GatingSignal = {}
