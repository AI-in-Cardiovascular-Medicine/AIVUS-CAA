import cv2

import numpy as np
from PyQt6.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPathItem,
    QPushButton,
    QCheckBox,
    QSizePolicy,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage, QColor, QPen, QBrush, QPainterPath

from tools.geometry import Marker


class LongitudinalView(QGraphicsView):
    """
    Displays the longitudinal view of the IVUS pullback with lumen area overlay.
    """

    DOT_RADIUS = 3
    MARGIN_TOP = 0.05  # fraction of image_height reserved at top
    MARGIN_BOTTOM = 0.05  # fraction of image_height reserved at bottom

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.graphics_scene = QGraphicsScene()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setScene(self.graphics_scene)

        self._area_items: list[QGraphicsEllipseItem] = []
        self._phase_line_items: list = []
        self._breathing_items: list = []
        self._current_marker = None
        self._areas_hidden = False
        self.num_frames = 0
        self.image_height = 0
        self.color = getattr(main_window.config.display, "color_contour", "green")

        self._peak_btn = QPushButton('Peak', self)
        self._peak_btn.setCheckable(True)
        self._peak_btn.setFixedSize(55, 22)
        self._peak_btn.setStyleSheet(
            'QPushButton{background:#333;color:#00dcc8;border:1px solid #00dcc8;font-size:10px}'
            'QPushButton:checked{background:#00dcc8;color:#000}'
        )

        self._valley_btn = QPushButton('Valley', self)
        self._valley_btn.setCheckable(True)
        self._valley_btn.setFixedSize(55, 22)
        self._valley_btn.setStyleSheet(
            'QPushButton{background:#333;color:#dc64dc;border:1px solid #dc64dc;font-size:10px}'
            'QPushButton:checked{background:#dc64dc;color:#000}'
        )

        self._delete_btn = QPushButton('Delete', self)
        self._delete_btn.setCheckable(True)
        self._delete_btn.setFixedSize(55, 22)
        self._delete_btn.setStyleSheet(
            'QPushButton{background:#333;color:#ff6060;border:1px solid #ff6060;font-size:10px}'
            'QPushButton:checked{background:#ff6060;color:#000}'
        )

        # reset all manual peak/valley edits back to the automatic detection
        self._reset_btn = QPushButton('Auto', self)
        self._reset_btn.setFixedSize(55, 22)
        self._reset_btn.setToolTip('Reset breathing peaks/valleys to automatic detection')
        self._reset_btn.setStyleSheet('QPushButton{background:#333;color:#ddd;border:1px solid #777;font-size:10px}')
        self._reset_btn.clicked.connect(self._on_reset_auto)

        # "has breathing artefact?" - when unchecked, the curve is dotted and the
        # Filtered viewer skips breathing correction (manual shuffle only).
        self._artefact_cb = QCheckBox('Breathing artefact', self)
        self._artefact_cb.setChecked(True)
        self._artefact_cb.setToolTip(
            'Uncheck when there is no breathing artefact: the curve is shown dotted '
            'and Filtered only shuffles frames (no breathing correction).'
        )
        self._artefact_cb.setStyleSheet('QCheckBox{color:#ddd;font-size:10px;background:transparent}')
        self._artefact_cb.adjustSize()
        self._artefact_cb.toggled.connect(self._on_artefact_toggled)

        # mutual exclusion across all three mode buttons
        self._peak_btn.clicked.connect(self._on_peak_btn_clicked)
        self._valley_btn.clicked.connect(self._on_valley_btn_clicked)
        self._delete_btn.clicked.connect(self._on_delete_btn_clicked)

    def _on_peak_btn_clicked(self):
        self._valley_btn.setChecked(False)
        self._delete_btn.setChecked(False)

    def _on_valley_btn_clicked(self):
        self._peak_btn.setChecked(False)
        self._delete_btn.setChecked(False)

    def _on_delete_btn_clicked(self):
        self._peak_btn.setChecked(False)
        self._valley_btn.setChecked(False)

    def _on_reset_auto(self):
        """Discard all manual peak/valley edits and return to automatic detection."""
        gs = self.main_window.runtime_data.gating_signal
        if gs is not None:
            gs.pop('breathing_manual_peaks', None)
            gs.pop('breathing_manual_valleys', None)
            gs['breathing_manual_mode'] = False
        self._peak_btn.setChecked(False)
        self._valley_btn.setChecked(False)
        self._delete_btn.setChecked(False)
        self.plot_areas()

    def _has_artefact(self) -> bool:
        gs = self.main_window.runtime_data.gating_signal
        return bool(gs.get('has_breathing_artefact', True)) if gs else True

    def _on_artefact_toggled(self, checked: bool):
        gs = self.main_window.runtime_data.gating_signal
        if gs is None:
            gs = {}
            self.main_window.runtime_data.gating_signal = gs
        gs['has_breathing_artefact'] = bool(checked)
        self.plot_areas()  # redraw curve solid/dotted

    def set_data(self, images):
        self.graphics_scene.clear()
        self._area_items = []
        self._breathing_items = []
        self._phase_line_items = []
        self.num_frames = images.shape[0]
        self.image_height = images.shape[1]
        center_col = images.shape[2] // 2

        if self.main_window.runtime_data.images_rgb is not None:
            slice_data = self.main_window.runtime_data.images_rgb[:, :, center_col, :]
        else:
            gray = self.main_window.runtime_data.images[:, :, center_col]  # (frames, height)
            if gray.dtype != np.uint8:
                # Normalize to uint8 using the current display windowing so the
                # longitudinal view matches what the main display shows.
                lo = self.main_window.display.window_level - self.main_window.display.window_width / 2
                hi = self.main_window.display.window_level + self.main_window.display.window_width / 2
                gray = np.clip(gray, lo, hi)
                span = hi - lo
                gray = ((gray - lo) / span * 255).astype(np.uint8) if span > 0 else np.zeros_like(gray, dtype=np.uint8)
            slice_data = np.stack([gray, gray, gray], axis=-1)  # (frames, height, 3)
        slice_data = np.transpose(slice_data, (1, 0, 2)).copy()
        q_format = QImage.Format.Format_RGB888
        bytes_per_line = self.num_frames * 3

        if getattr(self.main_window, 'colormap_enabled', False):
            gray_temp = cv2.cvtColor(slice_data, cv2.COLOR_RGB2GRAY)
            slice_data = cv2.applyColorMap(gray_temp, cv2.COLORMAP_COOL)
            slice_data = cv2.cvtColor(slice_data, cv2.COLOR_BGR2RGB)
            q_format = QImage.Format.Format_RGB888
            bytes_per_line = self.num_frames * 3

        longitudinal_image = QImage(slice_data.data, self.num_frames, self.image_height, bytes_per_line, q_format)
        pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(longitudinal_image))
        self.graphics_scene.addItem(pixmap_item)
        self.setSceneRect(pixmap_item.boundingRect())

        self.stretch_to_fit()
        # reflect the persisted breathing-artefact flag without triggering a redraw
        gs = self.main_window.runtime_data.gating_signal
        self._artefact_cb.blockSignals(True)
        self._artefact_cb.setChecked(bool(gs.get('has_breathing_artefact', True)) if gs else True)
        self._artefact_cb.blockSignals(False)
        self.plot_areas()

    def plot_areas(self):
        """Read lumen areas from main_window.runtime_data.frame_data_dct and draw one dot per frame."""
        for item in self._area_items:
            if item.scene() == self.graphics_scene:
                self.graphics_scene.removeItem(item)
        self._area_items = []

        if not self.main_window.runtime_data.frame_data_dct or self.image_height == 0:
            return

        areas: dict[int, float] = {}
        phases: dict[int, float] = {}
        for frame, fd in self.main_window.runtime_data.frame_data_dct.items():
            area = fd.lumen.measurements.area
            phase = fd.phase
            if area is not None and area > 0:
                areas[frame] = area
                phases[frame] = phase

        if not areas:
            return

        max_area = max(areas.values())
        usable_height = self.image_height * (1.0 - self.MARGIN_TOP - self.MARGIN_BOTTOM)
        top_offset = self.image_height * self.MARGIN_TOP
        r = self.DOT_RADIUS

        brush = QBrush(QColor(self.color))
        no_pen = QPen(Qt.PenStyle.NoPen)

        for frame, area in areas.items():
            y = top_offset + (1.0 - area / max_area) * usable_height
            item = QGraphicsEllipseItem(-r, -r, r * 2, r * 2)
            item.setPos(frame, y)
            item.setFlag(item.GraphicsItemFlag.ItemIgnoresTransformations)
            if phases[frame] == '-':
                new_brush = brush
            elif phases[frame] == 'D':
                new_brush = QBrush(QColor(*self.main_window.diastole_color))
            elif phases[frame] == 'S':
                new_brush = QBrush(QColor(*self.main_window.systole_color))
            else:
                new_brush = QBrush(QColor('orange'))
            item.setBrush(new_brush)
            item.setPen(no_pen)
            if self._areas_hidden:
                item.setVisible(False)
            self.graphics_scene.addItem(item)
            self._area_items.append(item)

        self.plot_breathing_signal(areas, max_area)

    def plot_breathing_signal(self, areas: dict[int, float], max_area: float):
        """Compute and overlay a respiratory envelope derived from the area dots.

        Detrends the area signal (removes the pullback taper), low-pass filters
        at ~2x the respiratory rate to isolate breathing, and marks peaks (cyan)
        and valleys (magenta).  Peaks/valleys can be edited with the Peak / Valley
        / Delete buttons; once the user edits, the markers become fully manual
        (seeded from the auto guess) so all of them are deletable.
        """
        for item in self._breathing_items:
            try:
                if item.scene() == self.graphics_scene:
                    self.graphics_scene.removeItem(item)
            except RuntimeError:
                pass
        self._breathing_items = []

        if len(areas) < 30 or self.image_height == 0:
            return

        frames_arr = np.array(sorted(areas.keys()), dtype=float)
        areas_arr = np.array([areas[int(f)] for f in frames_arr])

        from gating.breathing_pipeline import compute_breathing_signal, compute_breathing_phases

        rt = self.main_window.runtime_data
        gated_set = set(getattr(rt, 'gated_frames_dia', [])) | set(getattr(rt, 'gated_frames_sys', []))
        fs = float(rt.metadata.get('frame_rate', 30))
        gs = rt.gating_signal
        f_heart = gs.get('f_heart') if gs else None
        f_resp_override = gs.get('f_resp_override') if gs else None

        breathing = compute_breathing_signal(
            frames_arr,
            areas_arr,
            gated_frames=gated_set,
            fs=fs,
            f_heart=f_heart,
            f_resp_override=f_resp_override,
            cache=gs,
        )
        residual = breathing['residual']
        smoothed = breathing['smoothed']
        f_resp = breathing['f_resp']
        if gs is not None:
            gs['f_resp'] = f_resp
            gs['breathing_residual'] = residual.tolist()
            gs['breathing_frames'] = frames_arr.tolist()

        display_signal = np.mean(areas_arr) + smoothed
        if gs is not None:
            gs['breathing_display_signal'] = display_signal.tolist()

        osc_ptp = smoothed.max() - smoothed.min()
        breathing_detected = osc_ptp > 0.05 * max_area

        usable_height = self.image_height * (1.0 - self.MARGIN_TOP - self.MARGIN_BOTTOM)
        top_offset = self.image_height * self.MARGIN_TOP

        def to_y(v: float) -> float:
            return top_offset + (1.0 - v / max_area) * usable_height

        has_artefact = self._has_artefact()

        path = QPainterPath()
        path.moveTo(frames_arr[0], to_y(display_signal[0]))
        for i in range(1, len(frames_arr)):
            path.lineTo(frames_arr[i], to_y(display_signal[i]))
        curve_item = QGraphicsPathItem(path)
        curve_pen = QPen(QColor(0, 200, 220))
        curve_pen.setCosmetic(True)
        curve_pen.setWidth(0)
        # No breathing artefact -> draw the curve dotted (no correction will apply)
        if not has_artefact:
            curve_pen.setStyle(Qt.PenStyle.DotLine)
        curve_item.setPen(curve_pen)
        self.graphics_scene.addItem(curve_item)
        self._breathing_items.append(curve_item)

        # When there's no breathing artefact, peaks/valleys are irrelevant - the
        # dotted curve alone signals that Filtered will only shuffle, not correct.
        if not breathing_detected or not has_artefact:
            return

        # auto mode: show auto-detected peaks/valleys (hollow) + any manual (filled).
        # manual mode (user has edited): only the user's labels - every one filled
        # and deletable, and nothing hidden perturbs the phase.
        manual_mode = bool(gs.get('breathing_manual_mode', False)) if gs else False
        manual_peaks = list(gs.get('breathing_manual_peaks', [])) if gs else []
        manual_valleys = list(gs.get('breathing_manual_valleys', [])) if gs else []
        gap = max(15, int(fs))
        phase, peaks_idx, valleys_idx = compute_breathing_phases(
            smoothed,
            manual_peaks=manual_peaks,
            manual_valleys=manual_valleys,
            frames_arr=frames_arr,
            anchor_gap=gap,
            manual_only=manual_mode,
        )
        if gs is not None:
            gs['breathing_phase'] = phase.tolist()
            if not manual_mode:
                gs['breathing_auto_peaks'] = [int(frames_arr[i]) for i in peaks_idx if 0 <= i < len(frames_arr)]
                gs['breathing_auto_valleys'] = [int(frames_arr[i]) for i in valleys_idx if 0 <= i < len(frames_arr)]

        manual_idx = set()
        for f in manual_peaks + manual_valleys:
            manual_idx.add(int(np.argmin(np.abs(frames_arr - f))))

        r = self.DOT_RADIUS + 1
        no_brush = QBrush(Qt.BrushStyle.NoBrush)
        peak_brush = QBrush(QColor(0, 220, 200))
        valley_brush = QBrush(QColor(220, 100, 220))

        peak_pen = QPen(QColor(0, 220, 200))
        peak_pen.setCosmetic(True)
        peak_pen.setWidth(1)
        valley_pen = QPen(QColor(220, 100, 220))
        valley_pen.setCosmetic(True)
        valley_pen.setWidth(1)

        def _draw_marker(idx, pen, brush_color):
            item = QGraphicsEllipseItem(-r, -r, r * 2, r * 2)
            item.setPos(frames_arr[idx], to_y(display_signal[idx]))
            item.setFlag(item.GraphicsItemFlag.ItemIgnoresTransformations)
            item.setPen(pen)
            filled = manual_mode or (idx in manual_idx)
            item.setBrush(brush_color if filled else no_brush)
            self.graphics_scene.addItem(item)
            self._breathing_items.append(item)

        for pi in peaks_idx:
            if 0 <= pi < len(frames_arr):
                _draw_marker(int(pi), peak_pen, peak_brush)
        for vi in valleys_idx:
            if 0 <= vi < len(frames_arr):
                _draw_marker(int(vi), valley_pen, valley_brush)

    # ------------------------------------------------------------------
    # Peak / valley labelling (breathing curve only)
    # ------------------------------------------------------------------

    def _merge_gap(self) -> int:
        """Tolerance (in frames) for snapping / merging / deleting breathing labels."""
        fs = float(self.main_window.runtime_data.metadata.get('frame_rate', 30))
        return max(10, int(fs // 2))

    def _enter_manual_mode(self):
        """Switch breathing labels to fully-manual editing (seeded from auto)."""
        gs = self.main_window.runtime_data.gating_signal
        if gs is None:
            gs = {}
            self.main_window.runtime_data.gating_signal = gs
        if gs.get('breathing_manual_mode'):
            return gs
        mp = gs.setdefault('breathing_manual_peaks', [])
        mv = gs.setdefault('breathing_manual_valleys', [])
        if not mp and not mv:
            mp.extend(int(f) for f in gs.get('breathing_auto_peaks', []))
            mv.extend(int(f) for f in gs.get('breathing_auto_valleys', []))
        if mp or mv:  # don't flip into manual mode with nothing to edit
            gs['breathing_manual_mode'] = True
        return gs

    def _delete_nearest_manual_marker(self, clicked_x: float) -> bool:
        """Remove the closest breathing peak/valley label to *clicked_x*.

        Never touches lumen-area (segmentation) points.  Returns True if removed.
        """
        gs = self._enter_manual_mode()
        manual_peaks = gs.get('breathing_manual_peaks', [])
        manual_valleys = gs.get('breathing_manual_valleys', [])
        candidates = [(f, manual_peaks) for f in manual_peaks] + [(f, manual_valleys) for f in manual_valleys]
        if not candidates:
            return False
        frame, lst = min(candidates, key=lambda c: abs(c[0] - clicked_x))
        tol = max(self._merge_gap(), (self.num_frames or 1) // 25)
        if abs(frame - clicked_x) > tol:
            return False
        lst.remove(frame)
        if not manual_peaks and not manual_valleys:  # deleted everything -> back to auto
            gs['breathing_manual_mode'] = False
        self.plot_areas()
        return True

    def _add_manual_label(self, clicked_x: float, is_peak: bool) -> None:
        """Place a manual peak/valley at the clicked frame (breathing curve only)."""
        N = self.num_frames
        if N == 0:
            return
        clicked_frame = int(round(min(max(clicked_x, 0), N - 1)))
        gs = self._enter_manual_mode()
        manual_peaks = gs.setdefault('breathing_manual_peaks', [])
        manual_valleys = gs.setdefault('breathing_manual_valleys', [])
        gap = self._merge_gap()
        for lst in (manual_peaks, manual_valleys):
            for f in [f for f in lst if abs(f - clicked_frame) < gap]:
                lst.remove(f)
        (manual_peaks if is_peak else manual_valleys).append(clicked_frame)
        self.plot_areas()

    def contextMenuEvent(self, event):
        # Right-click = delete nearest breathing label (never a segmentation point).
        self._delete_nearest_manual_marker(self.mapToScene(event.pos()).x())
        event.accept()

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() != Qt.MouseButton.LeftButton:
            return

        clicked_x = self.mapToScene(event.pos()).x()
        if self._delete_btn.isChecked():
            self._delete_nearest_manual_marker(clicked_x)
        elif self._peak_btn.isChecked():
            self._add_manual_label(clicked_x, is_peak=True)
        elif self._valley_btn.isChecked():
            self._add_manual_label(clicked_x, is_peak=False)

    def hide_lview_contours(self):
        self._areas_hidden = True
        for item in self._area_items:
            item.setVisible(False)

    def show_lview_contours(self):
        self._areas_hidden = False
        for item in self._area_items:
            item.setVisible(True)

    def remove_contours(self, lower_limit, upper_limit):
        """Called when contours are deleted for a range; refresh area overlay."""
        self.plot_areas()

    def _update_phase_lines(self):
        for item in self._phase_line_items:
            try:
                if item.scene() == self.graphics_scene:
                    self.graphics_scene.removeItem(item)
            except RuntimeError:
                pass
        self._phase_line_items = []

        mw = self.main_window
        dia_frames = getattr(mw.runtime_data, 'gated_frames_dia', [])
        sys_frames = getattr(mw.runtime_data, 'gated_frames_sys', [])

        for frame, rgb in [(f, mw.diastole_color) for f in dia_frames] + [(f, mw.systole_color) for f in sys_frames]:
            color = QColor(*rgb)
            color.setAlpha(200)
            pen = QPen(color, (self.DOT_RADIUS / 2), Qt.PenStyle.DotLine)
            pen.setCosmetic(True)
            item = QGraphicsLineItem(frame, 0, frame, self.image_height)
            item.setPen(pen)
            self.graphics_scene.addItem(item)
            self._phase_line_items.append(item)

    def update_marker(self, frame):
        if self._current_marker is not None:
            try:
                if self._current_marker.scene() == self.graphics_scene:
                    self.graphics_scene.removeItem(self._current_marker)
            except RuntimeError:
                pass  # C++ object already deleted by Qt; just drop the reference
            self._current_marker = None

        self._update_phase_lines()

        self._current_marker = Marker(frame, 0, frame, self.image_height)
        self.graphics_scene.addItem(self._current_marker)

    def stretch_to_fit(self):
        if self.graphics_scene.items():
            self.fitInView(self.sceneRect(), Qt.AspectRatioMode.IgnoreAspectRatio)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.stretch_to_fit()
        self._peak_btn.move(4, 4)
        self._valley_btn.move(63, 4)
        self._delete_btn.move(122, 4)
        self._reset_btn.move(181, 4)
        self._artefact_cb.move(240, 5)
