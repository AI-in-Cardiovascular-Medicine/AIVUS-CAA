import bisect

from functools import partial
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QVBoxLayout,
    QSplitter,
    QPushButton,
    QButtonGroup,
    QWidget,
    QFrame,
    QLayout,
)

from domain.all_types import OCT_QUALITY_LABELS
from pages.intravascular.popup_windows.frame_range_dialog import FrameRangeDialog
from pages.intravascular.popup_windows.small_display import SmallDisplay
from pages.intravascular.utils.helpers import SplitterPane
from segmentation.segment import segment


class RightHalf:
    def __init__(self, main_window):
        self.main_window = main_window

        # Outer container — stays in the main splitter forever
        self.right_widget = SplitterPane()
        self.right_layout = QVBoxLayout(self.right_widget)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)

        # Slider connection for per-frame OCT updates (connected once, checks mode at runtime)
        main_window.display_slider.valueChanged.connect(partial(update_oct_display, main_window))

        # Build default (non-OCT) content
        self.content_widget = self._build_non_oct()
        self.right_layout.addWidget(self.content_widget)

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _build_non_oct(self):
        mw = self.main_window
        root = QWidget()
        vbox = QVBoxLayout(root)

        checkboxes = QHBoxLayout()
        checkboxes.addWidget(mw.diastolic_frame_box)
        checkboxes.addWidget(mw.systolic_frame_box)
        checkboxes.addWidget(mw.use_diastolic_button)
        compare_btn = QPushButton('Compare Frames')
        compare_btn.setToolTip('Open a small display to compare two frames')
        compare_btn.clicked.connect(partial(open_small_display, mw))
        checkboxes.addWidget(compare_btn)
        checkboxes.addWidget(mw.gating_display.toolbar)
        vbox.addLayout(checkboxes)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(mw.gating_display)
        splitter.addWidget(self._build_longitudinal_pane())
        gating_size = mw.gating_display.sizeHint().height()
        splitter.setSizes([gating_size, gating_size])
        splitter.setStretchFactor(0, mw.config.display.gating_display_stretch)
        splitter.setStretchFactor(1, mw.config.display.lview_display_stretch)
        vbox.addWidget(splitter)

        vbox.addLayout(self._build_lower_buttons())
        return root

    def _build_longitudinal_pane(self):
        """Longitudinal view with a Raw/Filtered mode selector column on its left.

        'Raw'      → images in acquisition order (default).
        'Filtered' → gated frames reordered into breathing-corrected anatomical
                     order (en bloc per cardiac cycle); the left-half slider then
                     scrolls gated frames in this sorted order.
        """
        mw = self.main_window
        pane = QWidget()
        hbox = QHBoxLayout(pane)
        hbox.setContentsMargins(0, 0, 0, 0)
        hbox.setSpacing(2)

        btn_col = QVBoxLayout()
        btn_col.setContentsMargins(2, 2, 2, 2)

        self.raw_btn = QPushButton('Raw')
        self.raw_btn.setCheckable(True)
        self.raw_btn.setChecked(True)
        self.raw_btn.setFixedWidth(70)
        self.raw_btn.setToolTip('Show frames in acquisition (pullback) order')

        self.filtered_btn = QPushButton('Filtered')
        self.filtered_btn.setCheckable(True)
        self.filtered_btn.setFixedWidth(70)
        self.filtered_btn.setToolTip('Reorder gated frames into breathing-corrected order (en bloc per cardiac cycle)')

        group = QButtonGroup(pane)
        group.setExclusive(True)
        group.addButton(self.raw_btn)
        group.addButton(self.filtered_btn)
        self._lview_mode_group = group  # keep reference

        self.raw_btn.clicked.connect(partial(set_longitudinal_mode, mw, 'raw'))
        self.filtered_btn.clicked.connect(partial(set_longitudinal_mode, mw, 'filtered'))

        btn_col.addWidget(self.raw_btn)
        btn_col.addWidget(self.filtered_btn)
        btn_col.addStretch(1)

        hbox.addLayout(btn_col)
        hbox.addWidget(mw.longitudinal_view, stretch=1)
        return pane

    def _build_oct(self):
        mw = self.main_window
        root = QWidget()
        vbox = QVBoxLayout(root)

        checkboxes = QHBoxLayout()
        checkboxes.addWidget(mw.tagged_frame_button)
        checkboxes.addWidget(mw.use_tagged_button)
        # compare_btn = QPushButton('Compare Frames')
        # compare_btn.setToolTip('Open a small display to compare two frames')
        # compare_btn.clicked.connect(partial(open_small_display, mw))
        # checkboxes.addWidget(compare_btn)
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        checkboxes.addWidget(separator)
        for label in OCT_QUALITY_LABELS:
            checkboxes.addWidget(mw.oct_quality_buttons[label])
        vbox.addLayout(checkboxes)

        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.addWidget(QWidget())  # empty top placeholder — add functionality later
        splitter.addWidget(mw.longitudinal_view)
        splitter.setStretchFactor(0, mw.config.display.gating_display_stretch)
        splitter.setStretchFactor(1, mw.config.display.lview_display_stretch)
        vbox.addWidget(splitter)

        vbox.addLayout(self._build_lower_buttons(oct=True))
        return root

    def _build_lower_buttons(self, oct=False):
        mw = self.main_window
        layout = QVBoxLayout()
        segment_button = QPushButton('Automatic Segmentation')
        segment_button.setToolTip('Run deep learning based segmentation of lumen')
        segment_button.clicked.connect(partial(segment, mw))
        if oct:
            right_button = QPushButton('Tag Frames by Distance')
            right_button.setToolTip('Tag frames at regular distance intervals within a frame range')
            right_button.clicked.connect(partial(tag_frames_by_distance, mw))
        else:
            right_button = QPushButton('Extract Diastolic and Systolic Frames')
            right_button.setToolTip('Extract diastolic and systolic images from pullback')
            right_button.clicked.connect(mw.gating_plot)
        command_buttons = QHBoxLayout()
        command_buttons.addWidget(segment_button)
        command_buttons.addWidget(right_button)
        layout.addLayout(command_buttons)
        layout.addLayout(QHBoxLayout())  # measures placeholder
        return layout

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_for_modality(self):
        self.right_layout.removeWidget(self.content_widget)
        self.content_widget.setParent(None)
        self.content_widget.deleteLater()

        if self.main_window.runtime_data.metadata.get('modality') == 'OCT':
            self.main_window.runtime_data.gated_frames = self.main_window.runtime_data.tagged_frames
            self.content_widget = self._build_oct()
        else:
            self.content_widget = self._build_non_oct()

        self.right_layout.addWidget(self.content_widget)

    def __call__(self):
        return self.right_widget


# ---------------------------------------------------------------------------
# OCT helpers
# ---------------------------------------------------------------------------


def tag_frames_by_distance(main_window):
    if not main_window.image_displayed:
        return
    dialog = FrameRangeDialog(main_window, step=True)
    dialog.setWindowTitle('Tag Frames by Distance')
    if not dialog.exec():
        return

    lower_limit, upper_limit = dialog.getInputs()

    if dialog.isStepByMm():
        step_mm = dialog.getStepMm()
        if step_mm <= 0.0:
            return
        speed = main_window.runtime_data.metadata['pullback_speed']  # mm/s
        frame_rate = main_window.runtime_data.metadata['frame_rate']  # frames/s
        step_frames = step_mm / (speed / frame_rate)
    else:
        step_frames = dialog.getStepFrames()
        if step_frames <= 0:
            return

    for idx in main_window.runtime_data.tagged_frames:
        main_window.runtime_data.frame_data_dct[idx].phase = '-'
    main_window.runtime_data.tagged_frames.clear()

    i = 0
    while True:
        idx = lower_limit + round(i * step_frames)
        if idx >= upper_limit:
            break
        if idx not in main_window.runtime_data.tagged_frames:
            bisect.insort_left(main_window.runtime_data.tagged_frames, idx)
        main_window.runtime_data.frame_data_dct[idx].phase = 'T'
        i += 1

    main_window.display.update_display()


def toggle_tagged_frame(main_window, state_true, drag=False):
    if main_window.image_displayed:
        frame = main_window.display_slider.value()
        if state_true:
            if frame not in main_window.runtime_data.tagged_frames:
                bisect.insort_left(main_window.runtime_data.tagged_frames, frame)
            main_window.runtime_data.frame_data_dct[frame].phase = 'T'
        else:
            try:
                main_window.runtime_data.tagged_frames.remove(frame)
            except ValueError:
                pass
            if main_window.runtime_data.frame_data_dct[frame].phase == 'T':
                main_window.runtime_data.frame_data_dct[frame].phase = '-'
        main_window.display.update_display()


def use_tagged(main_window):
    if main_window.image_displayed:
        main_window.runtime_data.gated_frames = main_window.runtime_data.tagged_frames


def set_oct_quality(main_window, label):
    if main_window.image_displayed:
        frame = main_window.display_slider.value()
        main_window.runtime_data.frame_data_dct[frame].quality = label


def update_oct_display(main_window, frame):
    """Update Tagged Frame checkbox and quality buttons when the slider moves (OCT only)."""
    if main_window.image_displayed and main_window.runtime_data.metadata.get('modality') == 'OCT':
        main_window.tagged_frame_button.blockSignals(True)
        main_window.tagged_frame_button.setChecked(frame in main_window.runtime_data.tagged_frames)
        main_window.tagged_frame_button.blockSignals(False)
        quality = main_window.runtime_data.frame_data_dct[frame].quality
        main_window.oct_quality_buttons[quality].setChecked(True)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def set_longitudinal_mode(main_window, mode):
    """Raw/Filtered toggle.

    'Filtered' opens the breathing-sorted paired viewer (diastole | systole in
    breathing-corrected order). 'Raw' just clears the status hint.
    """
    if not main_window.image_displayed:
        return
    if mode == 'filtered':
        from pages.intravascular.popup_windows.breathing_sort_viewer import BreathingSortViewer

        main_window.breathing_sort_viewer = BreathingSortViewer(main_window)
        main_window.breathing_sort_viewer.show()
    else:
        main_window.status_bar.showMessage(main_window.waiting_status)


def open_small_display(main_window):
    if main_window.image_displayed:
        main_window.small_display = SmallDisplay(main_window)
        main_window.small_display.move(
            main_window.x() + main_window.width() // 2, main_window.y() + main_window.height() // 2
        )
        next_gated = main_window.display_slider.next_gated_frame(set=False)
        main_window.small_display.update_frame(next_gated, update_image=True, update_contours=True, update_text=True)
        main_window.small_display.show()


def toggle_diastolic_frame(main_window, state_true, drag=False):
    if main_window.image_displayed:
        frame = main_window.display_slider.value()
        if state_true:
            main_window.use_diastolic_button.setChecked(True)
            use_diastolic(main_window)
            if frame not in main_window.runtime_data.gated_frames_dia:
                bisect.insort_left(main_window.runtime_data.gated_frames_dia, frame)
                main_window.runtime_data.frame_data_dct[frame].phase = 'D'
                main_window.gating_plot.update_color(main_window.diastole_color_plt)
                main_window.gating_plot.current_phase = 'D'
            try:  # frame cannot be diastolic and systolic at the same time
                main_window.systolic_frame_box.setChecked(False)
            except ValueError:
                pass
        else:
            try:
                main_window.runtime_data.gated_frames_dia.remove(frame)
                main_window.gating_plot.current_phase = None
                if (
                    main_window.runtime_data.frame_data_dct[frame].phase == 'D'
                ):  # do not reset when function is called from toggle_systolic_frame
                    main_window.runtime_data.frame_data_dct[frame].phase = '-'
                    if not drag:
                        main_window.gating_plot.update_color()
            except ValueError:
                pass

        main_window.display.update_display()


def toggle_systolic_frame(main_window, state_true, drag=False):
    if main_window.image_displayed:
        frame = main_window.display_slider.value()
        if state_true:
            main_window.use_diastolic_button.setChecked(False)
            use_diastolic(main_window)
            if frame not in main_window.runtime_data.gated_frames_sys:
                bisect.insort_left(main_window.runtime_data.gated_frames_sys, frame)
                main_window.runtime_data.frame_data_dct[frame].phase = 'S'
                main_window.gating_plot.update_color(main_window.systole_color_plt)
                main_window.gating_plot.current_phase = 'S'
            try:  # frame cannot be diastolic and systolic at the same time
                main_window.diastolic_frame_box.setChecked(False)
            except ValueError:
                pass
        else:
            try:
                main_window.runtime_data.gated_frames_sys.remove(frame)
                main_window.gating_plot.current_phase = None
                if (
                    main_window.runtime_data.frame_data_dct[frame].phase == 'S'
                ):  # do not reset when function is called from toggle_diastolic_frame
                    main_window.runtime_data.frame_data_dct[frame].phase = '-'
                    if not drag:
                        main_window.gating_plot.update_color()
            except ValueError:
                pass

        main_window.display.update_display()


def use_diastolic(main_window):
    if main_window.image_displayed:
        if main_window.use_diastolic_button.isChecked():
            main_window.use_diastolic_button.setText('Diastolic Frames')
            main_window.use_diastolic_button.setStyleSheet(f'background-color: rgb{main_window.diastole_color}')
            main_window.runtime_data.gated_frames = main_window.runtime_data.gated_frames_dia
        else:
            main_window.use_diastolic_button.setText('Systolic Frames')
            main_window.use_diastolic_button.setStyleSheet(f'background-color: rgb{main_window.systole_color}')
            main_window.runtime_data.gated_frames = main_window.runtime_data.gated_frames_sys

        try:
            next_gated = main_window.display_slider.next_gated_frame(set=False)
            main_window.small_display.update_frame(
                next_gated, update_image=True, update_contours=True, update_text=True
            )  # update small display
        except AttributeError:
            pass
