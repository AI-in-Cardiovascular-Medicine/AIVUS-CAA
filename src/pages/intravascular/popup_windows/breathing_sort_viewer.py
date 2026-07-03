"""Breathing-sorted paired viewer.

Opened from the "Filtered" button.  Runs the breathing-bin registration sort
(gating_pipeline.register_phase) independently for the gated diastolic and
systolic frames, then lets the user scroll through the breathing-corrected
order with a central slider.  Diastole and systole are shown side by side with
their lumen contours, each flanked by a 5-image filmstrip of its sorted-order
neighbours (index on top) that updates with the slider.  A swap tool
("switch index A with index B" + Apply) lets the user reorder outliers by hand.

Only gated frames are shown; gaps (positions with no frame) are ignored.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsPathItem,
    QSpinBox,
    QComboBox,
    QPushButton,
    QSlider,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage, QPen, QColor, QPainterPath
from loguru import logger

from gating.breathing_pipeline import (
    register_phase,
    assign_breathing_bins,
    compute_breathing_signal,
    compute_breathing_phases,
)
from tools.geometry import SplineGeometry

N_STRIP = 5  # thumbnails per filmstrip (odd → current one centred)
# Persistence: the sort (peaks/valleys, ordered dia/sys indices, per-bin shifts)
# is cached in runtime_data.gating_signal, which is written to / read from the
# _contours_*.json automatically.  Reused when anchors are unchanged; membership
# is reconciled when gated frames change; manual moves are stored immediately.


class BreathingSortViewer(QMainWindow):
    def __init__(self, main_window):
        super().__init__(main_window)
        self.main_window = main_window
        cfg = main_window.config.display
        self.image_size = cfg.image_size
        self.n_points_contour = cfg.n_points_contour
        self.contour_color = getattr(cfg, 'color_contour', 'green')
        images = main_window.runtime_data.images
        self.image_width = images.shape[1] if images is not None else self.image_size
        self.scaling = self.image_size / self.image_width
        self.thumb_size = int(self.image_size / 4)

        self.setWindowTitle('Breathing-sorted frames (diastole | systole)')

        self.dia_sorted: list[int] = []
        self.sys_sorted: list[int] = []
        self.dia_pos: dict[int, float] = {}
        self.sys_pos: dict[int, float] = {}
        self.dia_shifts: list[float] = []
        self.sys_shifts: list[float] = []
        self.peaks: list[int] = []
        self.valleys: list[int] = []
        self.n_bins: int = 4
        self.dia_idx = 0
        self.sys_idx = 0
        # index offset aligning the two phases at the ostium (most-proximal
        # reference point); None → fall back to nearest-corrected-position pairing
        self._anchor_offset = None

        self._compute_sort()
        self._build_ui()

        if not self.dia_sorted:
            self.info_label.setText(
                'Not enough gated frames / breathing labels to sort. '
                'Run gating and label a few breathing peaks & valleys first.'
            )
            self.slider.setEnabled(False)
            self.apply_btn.setEnabled(False)
            return

        self.slider.setRange(0, len(self.dia_sorted) - 1)
        self.slider.setValue(0)
        self._sync_swap_ranges()
        self._on_slider(0)

    # ------------------------------------------------------------------
    # Sort
    # ------------------------------------------------------------------
    def _areas(self):
        out = {}
        for frame, fd in self.main_window.runtime_data.frame_data_dct.items():
            a = fd.lumen.measurements.area
            if a is not None and a > 0:
                out[frame] = float(a)
        return out

    def _breathing_anchors(self, area_of):
        rt = self.main_window.runtime_data
        if rt.gating_signal is None:
            rt.gating_signal = {}
        gs = rt.gating_signal
        peaks = list(gs.get('breathing_manual_peaks') or gs.get('breathing_auto_peaks') or [])
        valleys = list(gs.get('breathing_manual_valleys') or gs.get('breathing_auto_valleys') or [])
        if peaks and valleys:
            return peaks, valleys
        if len(area_of) < 30:
            return peaks, valleys
        frames = np.array(sorted(area_of.keys()), dtype=float)
        areas = np.array([area_of[int(f)] for f in frames])
        fs = float(rt.metadata.get('frame_rate', 30))
        breathing = compute_breathing_signal(frames, areas, fs=fs, f_heart=gs.get('f_heart'), cache=gs)
        _, pk, vl = compute_breathing_phases(breathing['smoothed'], frames_arr=frames)
        peaks = [int(frames[i]) for i in pk if 0 <= i < len(frames)]
        valleys = [int(frames[i]) for i in vl if 0 <= i < len(frames)]
        return peaks, valleys

    def _signature(self, peaks, valleys, n_bins):
        """Identity of the anchors a sort was built from (to detect changes)."""
        return {
            'peaks': sorted(int(p) for p in peaks),
            'valleys': sorted(int(v) for v in valleys),
            'n_bins': int(n_bins),
        }

    def _compute_sort(self):
        """Reuse the stored sort when the breathing anchors are unchanged; else
        recompute.  Either way, reconcile membership with the current gated sets
        (drop removed frames, insert newly-gated ones) so manual ordering is kept.
        """
        rt = self.main_window.runtime_data
        if rt.gating_signal is None:
            rt.gating_signal = {}
        gs = rt.gating_signal

        area_of = self._areas()
        self.n_bins = int(getattr(self.main_window.config.gating, 'breathing_bins', 4))
        self.peaks, self.valleys = self._breathing_anchors(area_of)
        n_total = int(rt.metadata.get('num_frames', 0)) or (len(rt.images) if rt.images is not None else 0)

        dia = [f for f in sorted(rt.gated_frames_dia) if f in area_of]
        sys = [f for f in sorted(rt.gated_frames_sys) if f in area_of]
        if len(dia) < 5:
            logger.info('Breathing sort: too few diastolic frames with contours')
            return

        sig = self._signature(self.peaks, self.valleys, self.n_bins)
        cached = gs.get('sort_signature') == sig and gs.get('sort_dia_order') is not None

        if cached:
            logger.info('Breathing sort: reusing stored order (anchors unchanged)')
            self.dia_sorted = [int(f) for f in gs.get('sort_dia_order', [])]
            self.sys_sorted = [int(f) for f in gs.get('sort_sys_order', [])]
            self.dia_pos = {int(f): float(p) for f, p in gs.get('sort_dia_pos', [])}
            self.sys_pos = {int(f): float(p) for f, p in gs.get('sort_sys_pos', [])}
            self.dia_shifts = [float(s) for s in gs.get('sort_dia_shifts', [])]
            self.sys_shifts = [float(s) for s in gs.get('sort_sys_shifts', [])]
            self.dia_sorted = self._reconcile(self.dia_sorted, dia, self.dia_pos, self.dia_shifts)
            self.sys_sorted = self._reconcile(self.sys_sorted, sys, self.sys_pos, self.sys_shifts)
        else:
            logger.info('Breathing sort: recomputing (anchors changed / no cache)')
            self._recompute(dia, sys, area_of, n_total)

        if self.dia_sorted:
            self._store_sort()
        self._update_anchor()

    def _update_anchor(self):
        """Compute the dia→sys index offset that lines up the two ostium
        (most-proximal, highest-frame-index) reference points, one per phase.

        Pairing by this fixed offset keeps diastole and systole aligned even when
        the two phases have different frame counts (e.g. ostium has only
        diastole).  None if either phase has no reference point → caller falls
        back to nearest-corrected-position pairing.
        """
        self._anchor_offset = None
        if not self.dia_sorted or not self.sys_sorted:
            return
        fdd = self.main_window.runtime_data.frame_data_dct or {}
        dia_refs = [f for f in self.dia_sorted if fdd.get(f) is not None and fdd[f].reference is not None]
        sys_refs = [f for f in self.sys_sorted if fdd.get(f) is not None and fdd[f].reference is not None]
        if not dia_refs or not sys_refs:
            return
        dia_ostium = max(dia_refs)  # highest frame index = most proximal = ostium
        sys_ostium = max(sys_refs)
        self._anchor_offset = self.sys_sorted.index(sys_ostium) - self.dia_sorted.index(dia_ostium)
        logger.info(f'Breathing sort: ostium anchor offset dia→sys = {self._anchor_offset}')

    def _recompute(self, dia, sys, area_of, n_total):
        Rd = register_phase(
            np.array(dia, float),
            np.array([area_of[f] for f in dia], float),
            self.peaks,
            self.valleys,
            n_bins=self.n_bins,
            n_total=n_total,
        )
        self.dia_sorted = [dia[i] for i in Rd['order']]
        self.dia_pos = {dia[i]: float(Rd['corrected'][i]) for i in range(len(dia))}
        self.dia_shifts = [float(s) for s in Rd['shifts']]

        self.sys_sorted, self.sys_pos, self.sys_shifts = [], {}, []
        if len(sys) >= 5:
            Rs = register_phase(
                np.array(sys, float),
                np.array([area_of[f] for f in sys], float),
                self.peaks,
                self.valleys,
                n_bins=self.n_bins,
                n_total=n_total,
            )
            self.sys_sorted = [sys[i] for i in Rs['order']]
            self.sys_pos = {sys[i]: float(Rs['corrected'][i]) for i in range(len(sys))}
            self.sys_shifts = [float(s) for s in Rs['shifts']]

    def _reconcile(self, order, current_frames, pos_map, shifts):
        """Drop frames no longer gated and insert newly-gated ones by corrected
        position, preserving the existing (possibly hand-edited) order otherwise."""
        current = set(current_frames)
        new_order = [f for f in order if f in current]
        existing = set(new_order)
        for f in current_frames:
            if f in existing:
                continue
            # corrected position of the new frame from its bin's stored shift
            b = int(assign_breathing_bins(np.array([f]), self.peaks, self.valleys, self.n_bins)[0])
            shift = shifts[b] if 0 <= b < len(shifts) else 0.0
            pos = float(f) + float(shift)
            pos_map[f] = pos
            insert_at = len(new_order)
            for idx, g in enumerate(new_order):
                if pos < pos_map.get(g, float(g)):
                    insert_at = idx
                    break
            new_order.insert(insert_at, f)
        return new_order

    def _store_sort(self):
        """Persist the current sort into gating_signal (auto-saved to the JSON)."""
        gs = self.main_window.runtime_data.gating_signal
        gs['sort_signature'] = self._signature(self.peaks, self.valleys, self.n_bins)
        gs['sort_peaks'] = sorted(int(p) for p in self.peaks)
        gs['sort_valleys'] = sorted(int(v) for v in self.valleys)
        gs['sort_n_bins'] = int(self.n_bins)
        gs['sort_dia_order'] = [int(f) for f in self.dia_sorted]
        gs['sort_sys_order'] = [int(f) for f in self.sys_sorted]
        gs['sort_dia_pos'] = [[int(f), float(p)] for f, p in self.dia_pos.items()]
        gs['sort_sys_pos'] = [[int(f), float(p)] for f, p in self.sys_pos.items()]
        gs['sort_dia_shifts'] = [float(s) for s in self.dia_shifts]
        gs['sort_sys_shifts'] = [float(s) for s in self.sys_shifts]

    def _nearest_sys(self, corrected_pos: float) -> int:
        if not self.sys_sorted:
            return 0
        arr = np.array([self.sys_pos[f] for f in self.sys_sorted])
        return int(np.argmin(np.abs(arr - corrected_pos)))

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        central = QWidget()
        vbox = QVBoxLayout(central)

        row = QHBoxLayout()
        self.dia_strip = self._make_filmstrip('dia')
        self.sys_strip = self._make_filmstrip('sys')

        self.dia_scene = QGraphicsScene()
        self.dia_view = QGraphicsView(self.dia_scene)
        self.dia_pixmap = QGraphicsPixmapItem()
        self.dia_scene.addItem(self.dia_pixmap)

        self.sys_scene = QGraphicsScene()
        self.sys_view = QGraphicsView(self.sys_scene)
        self.sys_pixmap = QGraphicsPixmapItem()
        self.sys_scene.addItem(self.sys_pixmap)

        dia_col = QVBoxLayout()
        self.dia_title = QLabel('Diastole')
        self.dia_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.dia_title.setStyleSheet('color:white;font-weight:bold;')
        dia_col.addWidget(self.dia_title)
        dia_col.addWidget(self.dia_view)

        sys_col = QVBoxLayout()
        self.sys_title = QLabel('Systole')
        self.sys_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sys_title.setStyleSheet('color:white;font-weight:bold;')
        sys_col.addWidget(self.sys_title)
        sys_col.addWidget(self.sys_view)

        row.addWidget(self.dia_strip['widget'])
        row.addLayout(dia_col, stretch=2)
        row.addLayout(sys_col, stretch=2)
        row.addWidget(self.sys_strip['widget'])
        vbox.addLayout(row, stretch=1)

        # central slider (main navigation through the sorted slots)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self._on_slider)
        vbox.addWidget(self.slider)

        self.info_label = QLabel('')
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setStyleSheet('color:white;')
        vbox.addWidget(self.info_label)

        # swap tool: "switch index A with index B" + Apply, for the chosen phase
        swap = QGridLayout()
        swap.addWidget(QLabel('Phase:'), 0, 0)
        self.phase_combo = QComboBox()
        self.phase_combo.addItems(['Diastole', 'Systole'])
        self.phase_combo.currentIndexChanged.connect(lambda _: self._sync_swap_ranges())
        swap.addWidget(self.phase_combo, 0, 1)

        swap.addWidget(QLabel('Move index:'), 1, 0)
        self.swap_a = QSpinBox()
        swap.addWidget(self.swap_a, 1, 1)
        swap.addWidget(QLabel('to position:'), 2, 0)
        self.swap_b = QSpinBox()
        swap.addWidget(self.swap_b, 2, 1)

        self.apply_btn = QPushButton('Apply move')
        self.apply_btn.clicked.connect(self._apply_move)
        swap.addWidget(self.apply_btn, 3, 0, 1, 2)

        swap_box = QHBoxLayout()
        swap_box.addStretch(1)
        swap_box.addLayout(swap)
        swap_box.addStretch(1)
        vbox.addLayout(swap_box)

        self.setCentralWidget(central)
        self.resize(int(self.image_size * 2.2), int(self.image_size * 1.2))

    def _make_filmstrip(self, which: str) -> dict:
        w = QWidget()
        col = QVBoxLayout(w)
        col.setContentsMargins(2, 2, 2, 2)
        cells = []
        for k in range(N_STRIP):
            cell = QVBoxLayout()
            idx_lab = QLabel('')
            idx_lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
            idx_lab.setStyleSheet('color:#aaa;font-size:9px;')
            img_lab = QLabel()
            img_lab.setFixedSize(self.thumb_size, self.thumb_size)
            img_lab.setStyleSheet('border:1px solid #333;')
            img_lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_lab.mousePressEvent = lambda ev, wc=which, kk=k: self._on_thumb_click(wc, kk)  # type: ignore[method-assign,misc]
            cell.addWidget(idx_lab)
            cell.addWidget(img_lab)
            col.addLayout(cell)
            cells.append((idx_lab, img_lab))
        col.addStretch(1)
        return {'widget': w, 'cells': cells}

    def _sync_swap_ranges(self):
        seq = self.dia_sorted if self.phase_combo.currentText() == 'Diastole' else self.sys_sorted
        hi = max(0, len(seq) - 1)
        for sp in (self.swap_a, self.swap_b):
            sp.setRange(0, hi)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _grayscale(self, frame):
        images = self.main_window.runtime_data.images
        img = images[frame]
        if img.dtype != np.uint8:
            lo = self.main_window.display.window_level - self.main_window.display.window_width / 2
            hi = self.main_window.display.window_level + self.main_window.display.window_width / 2
            span = hi - lo
            img = np.clip(img, lo, hi)
            img = ((img - lo) / span * 255).astype(np.uint8) if span > 0 else np.zeros_like(img, dtype=np.uint8)
        return np.ascontiguousarray(img)

    def _pixmap(self, frame, size):
        if frame is None or self.main_window.runtime_data.images is None:
            return QPixmap()
        img = self._grayscale(frame)
        h, w = img.shape
        qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8).scaled(
            size, size, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation
        )
        return QPixmap.fromImage(qimg)

    def _draw_main(self, scene, pixmap_item, view, frame):
        for it in list(scene.items()):
            if isinstance(it, QGraphicsPathItem) and it.scene() == scene:
                scene.removeItem(it)
        if frame is None:
            pixmap_item.setPixmap(QPixmap())
            return
        pixmap_item.setPixmap(self._pixmap(frame, self.image_size))
        self._draw_contour(scene, frame)
        scene.setSceneRect(0, 0, self.image_size, self.image_size)
        view.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _draw_contour(self, scene, frame):
        fd = self.main_window.runtime_data.frame_data_dct.get(frame)
        if fd is None:
            return
        lumen = getattr(fd, 'lumen', None)
        if lumen is None or not lumen.contours or not lumen.contours[0] or not lumen.contours[0][0]:
            return
        xs = [x * self.scaling for x in lumen.contours[0][0]]
        ys = [y * self.scaling for y in lumen.contours[0][1]]
        if len(xs) < 3:
            return
        fx: np.ndarray | list
        fy: np.ndarray | list
        try:
            geo = SplineGeometry(xs, ys, self.n_points_contour, None, None)
            geo.interpolate()
            fx, fy = geo.full_contour[0], geo.full_contour[1]
            if fx is None:
                fx, fy = xs, ys
        except Exception:
            fx, fy = xs, ys
        path = QPainterPath()
        path.moveTo(float(fx[0]), float(fy[0]))
        for i in range(1, len(fx)):
            path.lineTo(float(fx[i]), float(fy[i]))
        path.closeSubpath()
        item = QGraphicsPathItem(path)
        pen = QPen(QColor(self.contour_color))
        pen.setWidth(2)
        item.setPen(pen)
        scene.addItem(item)

    def _update_strip(self, strip, sorted_list, current_idx):
        half = N_STRIP // 2
        for k, (idx_lab, img_lab) in enumerate(strip['cells']):
            j = current_idx - half + k
            if 0 <= j < len(sorted_list):
                frame = sorted_list[j]
                tag = f'[{j}] f{frame + 1}'
                if k == half:
                    tag = f'► {tag}'
                idx_lab.setText(tag)
                img_lab.setPixmap(self._pixmap(frame, self.thumb_size))
            else:
                idx_lab.setText('')
                img_lab.setPixmap(QPixmap())

    # ------------------------------------------------------------------
    # Interaction
    # ------------------------------------------------------------------
    def _on_slider(self, v):
        self.dia_idx = v
        if self.dia_sorted and self.sys_sorted:
            if self._anchor_offset is not None:
                # align at the ostium reference: fixed offset from the anchor
                self.sys_idx = max(0, min(len(self.sys_sorted) - 1, v + self._anchor_offset))
            else:
                self.sys_idx = self._nearest_sys(self.dia_pos[self.dia_sorted[v]])
        # keep swap "A" pointing at the current slot of the selected phase
        if self.phase_combo.currentText() == 'Diastole':
            self.swap_a.setValue(min(v, self.swap_a.maximum()))
        else:
            self.swap_a.setValue(min(self.sys_idx, self.swap_a.maximum()))
        self._refresh()

    def _on_thumb_click(self, which, cell_k):
        half = N_STRIP // 2
        if which == 'dia':
            j = self.dia_idx - half + cell_k
            if 0 <= j < len(self.dia_sorted):
                self.slider.setValue(j)  # move main navigation
        else:
            j = self.sys_idx - half + cell_k
            if 0 <= j < len(self.sys_sorted):
                self.sys_idx = j  # manual systole override for this slot
                self._refresh()

    def _apply_move(self):
        """Move the frame at index A to position B; the rest keep their order."""
        phase = self.phase_combo.currentText()
        seq = self.dia_sorted if phase == 'Diastole' else self.sys_sorted
        a, b = self.swap_a.value(), self.swap_b.value()
        if a == b or not (0 <= a < len(seq)) or not (0 <= b < len(seq)):
            return
        frame = seq.pop(a)
        seq.insert(b, frame)
        self._store_sort()  # persist the manual reordering
        self._update_anchor()  # ostium offset may have shifted after reordering
        # follow the moved frame if we're viewing that phase
        if phase == 'Diastole':
            self.slider.setValue(b)
        else:
            self.sys_idx = b
        self._refresh()

    def _refresh(self):
        if not self.dia_sorted:
            return
        dia_frame = self.dia_sorted[self.dia_idx] if 0 <= self.dia_idx < len(self.dia_sorted) else None
        sys_frame = self.sys_sorted[self.sys_idx] if 0 <= self.sys_idx < len(self.sys_sorted) else None

        self._draw_main(self.dia_scene, self.dia_pixmap, self.dia_view, dia_frame)
        self._draw_main(self.sys_scene, self.sys_pixmap, self.sys_view, sys_frame)
        self._update_strip(self.dia_strip, self.dia_sorted, self.dia_idx)
        self._update_strip(self.sys_strip, self.sys_sorted, self.sys_idx)

        self.dia_title.setText(f'Diastole — frame {dia_frame + 1}' if dia_frame is not None else 'Diastole')
        self.sys_title.setText(f'Systole — frame {sys_frame + 1}' if sys_frame is not None else 'Systole — (none)')
        dp = self.dia_pos.get(dia_frame) if dia_frame is not None else None
        sp = self.sys_pos.get(sys_frame) if sys_frame is not None else None
        txt = f'Slot {self.dia_idx + 1}/{len(self.dia_sorted)}'
        if dp is not None:
            txt += f'   dia pos ≈ {dp:.0f}'
        if sp is not None:
            txt += f'   sys pos ≈ {sp:.0f}'
        self.info_label.setText(txt)
