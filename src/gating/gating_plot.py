import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.widgets import Button
from PyQt6.QtWidgets import QMainWindow, QVBoxLayout, QWidget

import gating.gating_pipeline as pipeline
from gating.automatic_gating import AutomaticGating
from pages.intravascular.popup_windows.message_boxes import ErrorMessage
from pages.intravascular.popup_windows.frame_range_dialog import FrameRangeDialog
from pages.intravascular.right_half.right_half import toggle_diastolic_frame, toggle_systolic_frame
from input_output.output.reports import report

logging.getLogger('matplotlib').setLevel(logging.WARNING)


class GatingPlot:
    def __init__(self, main_window):
        self.main_window = main_window
        self.vertical_lines = []
        self.selected_line = None
        self.current_phase = None
        self.tmp_phase = None
        self.frame_marker = None
        self.lower_limit = 0
        self.default_line_color = 'grey'
        self.default_linestyle = (0, (1, 3))

    def _draw(self):
        try:
            self.fig.canvas.draw_idle()
        except AttributeError:
            pass

    @property
    def _ready(self) -> bool:
        return hasattr(self, 'ax') and hasattr(self, 'fig')

    def _toolbar_active(self) -> bool:
        toolbar = getattr(self.main_window.gating_display, 'toolbar', None)
        return toolbar is not None and bool(toolbar.mode)

    def __call__(self):
        self.main_window.status_bar.showMessage('Gating…')
        dialog_success = self.define_roi()
        if not dialog_success:
            self.main_window.status_bar.showMessage(self.main_window.waiting_status)
            return
        (
            image_based_gating,
            contour_based_gating,
            image_based_gating_filtered,
            contour_based_gating_filtered,
        ) = pipeline.prepare_data(
            self.main_window,
            self.frames,
            self.report_data,
            lower_limit=self.lower_limit,
        )
        self.plot_data(
            image_based_gating,
            contour_based_gating,
            image_based_gating_filtered,
            contour_based_gating_filtered,
        )
        self.main_window.status_bar.showMessage(self.main_window.waiting_status)

    def define_roi(self):
        dialog = FrameRangeDialog(self.main_window)
        if not dialog.exec():
            return False

        lower_limit, upper_limit = dialog.getInputs()
        self.lower_limit = lower_limit

        # Try to get report data; gating does NOT require all frames to have contours
        self.report_data = report(self.main_window, lower_limit, upper_limit, suppress_messages=True)

        if self.report_data is None:
            # report() returns None either when no images are loaded or when the
            # runtime data is completely empty.  Check if the problem is "no images".
            if not self.main_window.image_displayed:
                ErrorMessage(self.main_window, 'Please ensure that an input file was read')
                return False
            # No contours at all - proceed with image-only gating
            self.report_data = pd.DataFrame()

        # Warn (don't block) when only a subset of frames has contours
        n_range = upper_limit - lower_limit
        if len(self.report_data) < n_range:
            coverage = len(self.report_data) / n_range if n_range > 0 else 0
            if coverage < 0.5:
                self.main_window.status_bar.showMessage(
                    f'Gating: only {coverage:.0%} of frames have contours - image-only mode'
                )

        self.frames = self.main_window.runtime_data.images[lower_limit:upper_limit]
        # x covers ALL frames in the range (1-indexed), not just contoured ones
        self.x = np.arange(lower_limit + 1, upper_limit + 1)
        return True

    def plot_data(
        self,
        image_based_gating,
        contour_based_gating,
        image_based_gating_filtered,
        contour_based_gating_filtered,
    ):
        # Shift unfiltered signals below the filtered ones (work on copies - originals
        # are still referenced by prepare_data's return values)
        image_based_gating = image_based_gating.copy()
        contour_based_gating = contour_based_gating.copy()
        min_sig = min(np.nanmin(image_based_gating), np.nanmin(contour_based_gating))
        for sig in (image_based_gating, contour_based_gating):
            sig += min_sig - np.nanmax(sig)

        self.fig = self.main_window.gating_display.fig
        self.fig.clear()
        self.frame_marker = None
        self.ax = self.fig.add_subplot()

        # Filtered signals - primary visual focus
        (self._img_line,) = self.ax.plot(
            self.x, image_based_gating_filtered, color='green', lw=2, label='Image (filtered)'
        )
        (self._cnt_line,) = self.ax.plot(
            self.x, contour_based_gating_filtered, color='yellow', lw=2, label='Contour (filtered)'
        )
        # Raw signals - subtle background reference
        self.ax.plot(self.x, image_based_gating, color='green', ls='dashed', alpha=0.30)
        self.ax.plot(self.x, contour_based_gating, color='yellow', ls='dashed', alpha=0.30)

        self.ax.set_xlabel('Frame')
        self.ax.get_yaxis().set_visible(False)
        legend = self.ax.legend(ncol=2, loc='lower right')
        legend.set_draggable(True)

        # Frequency-sweep buttons (bottom-left corner)
        ax_btn = self.fig.add_axes([0.01, 0.01, 0.12, 0.06])
        self._sweep_btn = Button(ax_btn, 'Freq. sweep', color='#333333', hovercolor='#555555')
        self._sweep_btn.label.set_color('white')
        self._sweep_btn.on_clicked(self._show_frequency_sweep)

        ax_btn2 = self.fig.add_axes([0.14, 0.01, 0.14, 0.06])
        self._breathing_sweep_btn = Button(ax_btn2, 'Breathing sweep', color='#333333', hovercolor='#555555')
        self._breathing_sweep_btn.label.set_color('white')
        self._breathing_sweep_btn.on_clicked(self._show_breathing_frequency_sweep)

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            self.fig.tight_layout()

        self._draw()

        self.draw_existing_lines(
            self.main_window.runtime_data.gated_frames_dia,
            self.main_window.diastole_color_plt,
        )
        self.draw_existing_lines(
            self.main_window.runtime_data.gated_frames_sys,
            self.main_window.systole_color_plt,
        )

        if not self.main_window.runtime_data.gated_frames_dia and not self.main_window.runtime_data.gated_frames_sys:
            auto_gating = AutomaticGating(self.main_window, self.report_data, self.lower_limit)
            auto_gating.automatic_gating(image_based_gating_filtered, contour_based_gating_filtered)
            self.draw_existing_lines(
                self.main_window.runtime_data.gated_frames_dia,
                self.main_window.diastole_color_plt,
            )
            self.draw_existing_lines(
                self.main_window.runtime_data.gated_frames_sys,
                self.main_window.systole_color_plt,
            )
            self._draw()

        return True

    def _show_frequency_sweep(self, *_):
        """Heatmap: image signal low-pass filtered at increasing BPM cutoffs.

        Click a row to interactively apply that cutoff to the main gating plot.
        The yellow line marks the currently active cutoff.
        """
        gs = getattr(self.main_window.runtime_data, 'gating_signal', None)
        if not gs or 'freq_sweep_signals' not in gs:
            return

        bpm_cuts = np.array(gs['freq_sweep_bpm_cuts'])
        sweep = np.array(gs['freq_sweep_signals'])
        f_heart = gs.get('f_heart', 1.0)
        image_raw = np.array(gs['image_based_gating'])
        fs = self.main_window.runtime_data.metadata['frame_rate']
        cfg = self.main_window.config.gating

        hr_bpm = f_heart * 60
        initial_bpm = 2.0 * hr_bpm  # yellow line starts at 2xHR

        sweep_fig, ax_sw = plt.subplots(figsize=(13, 5))
        sweep_fig.patch.set_facecolor('#1e1e1e')
        ax_sw.set_facecolor('#1e1e1e')

        im = ax_sw.pcolormesh(self.x, bpm_cuts, sweep, cmap='RdBu_r', shading='auto')

        # Active yellow line - starts at 2xHR, moves on click (legend index 0)
        (active_line,) = ax_sw.plot(
            [self.x[0], self.x[-1]],
            [initial_bpm, initial_bpm],
            color='yellow',
            lw=2,
            label=f'LP cutoff: {initial_bpm:.0f} BPM',
        )
        # 1xHR reference line - moves on click to show implied HR (legend index 1)
        (hr_line,) = ax_sw.plot(
            [self.x[0], self.x[-1]],
            [hr_bpm, hr_bpm],
            color='lime',
            lw=1.2,
            ls='--',
            label=f'HR = {hr_bpm:.0f} BPM',
        )
        ax_sw.axhline(initial_bpm, color='#888888', lw=1.0, ls=':', label=f'Initial 2xHR = {initial_bpm:.0f} BPM')

        ax_sw.set_xlabel('Frame', color='white')
        ax_sw.set_ylabel('Low-pass cutoff (BPM)', color='white')
        ax_sw.set_title('Frequency sweep - click a row to apply that cutoff', color='white')
        ax_sw.tick_params(colors='white')
        for sp in ax_sw.spines.values():
            sp.set_edgecolor('#555')
        legend = ax_sw.legend(loc='upper right', facecolor='#333', labelcolor='white')
        sweep_fig.colorbar(im, ax=ax_sw, label='Normalised amplitude')
        sweep_fig.tight_layout()

        def on_sweep_click(ev):
            if ev.inaxes != ax_sw or ev.ydata is None:
                return
            new_bpm = float(np.clip(ev.ydata, bpm_cuts[0], bpm_cuts[-1]))
            new_hz = new_bpm / 60.0

            implied_hr = new_bpm / 2.0
            active_line.set_ydata([new_bpm, new_bpm])
            hr_line.set_ydata([implied_hr, implied_hr])
            legend.get_texts()[0].set_text(f'LP cutoff: {new_bpm:.0f} BPM')
            legend.get_texts()[1].set_text(f'HR = {implied_hr:.0f} BPM')
            sweep_fig.canvas.draw_idle()

            new_filt = pipeline.lowpass_filter(image_raw, new_hz, fs)
            new_filt = pipeline.normalize_data(new_filt, step=cfg.normalize_step)
            self._img_line.set_ydata(new_filt)
            self.fig.canvas.draw_idle()

        canvas = FigureCanvasQTAgg(sweep_fig)
        canvas.mpl_connect('button_press_event', on_sweep_click)

        win = QMainWindow(self.main_window)
        win.setWindowTitle('Frequency sweep')
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas)
        win.setCentralWidget(container)
        win.resize(1300, 500)
        win.show()
        self._sweep_win = win  # keep reference so Qt GC doesn't destroy it

    def _show_breathing_frequency_sweep(self, *_):
        """Heatmap of area-residual low-pass filtered at increasing BrPM cutoffs.

        Click a row to set that cutoff as the breathing filter and refresh the
        longitudinal view overlay immediately.  The yellow line tracks the
        currently active cutoff.
        """
        gs = getattr(self.main_window.runtime_data, 'gating_signal', None)
        if not gs or 'breathing_residual' not in gs:
            return

        residual = np.array(gs['breathing_residual'])
        frames = np.array(gs['breathing_frames'])
        f_resp_current = gs.get('f_resp', 0.3)
        fs = self.main_window.runtime_data.metadata['frame_rate']

        brpm_cuts, sweep = pipeline.compute_frequency_sweep(residual, fs, bpm_lo=10.0, bpm_hi=60.0)
        current_brpm = f_resp_current * 60

        sweep_fig, ax_sw = plt.subplots(figsize=(13, 5))
        sweep_fig.patch.set_facecolor('#1e1e1e')
        ax_sw.set_facecolor('#1e1e1e')

        ax_sw.pcolormesh(frames, brpm_cuts, sweep, cmap='RdBu_r', shading='auto')

        (active_line,) = ax_sw.plot(
            [frames[0], frames[-1]],
            [current_brpm, current_brpm],
            color='yellow',
            lw=2,
            label=f'LP cutoff: {current_brpm:.1f} BrPM',
        )
        ax_sw.axhline(current_brpm, color='#888888', lw=1.0, ls=':', label=f'Auto-detected: {current_brpm:.1f} BrPM')

        ax_sw.set_xlabel('Frame', color='white')
        ax_sw.set_ylabel('Low-pass cutoff (BrPM)', color='white')
        ax_sw.set_title('Breathing frequency sweep — click a row to apply', color='white')
        ax_sw.tick_params(colors='white')
        for sp in ax_sw.spines.values():
            sp.set_edgecolor('#555')
        legend = ax_sw.legend(loc='upper right', facecolor='#333', labelcolor='white')
        sweep_fig.tight_layout()

        def on_sweep_click(ev):
            if ev.inaxes != ax_sw or ev.ydata is None:
                return
            new_brpm = float(np.clip(ev.ydata, brpm_cuts[0], brpm_cuts[-1]))
            new_hz = new_brpm / 60.0
            active_line.set_ydata([new_brpm, new_brpm])
            legend.get_texts()[0].set_text(f'LP cutoff: {new_brpm:.1f} BrPM')
            sweep_fig.canvas.draw_idle()
            gs['f_resp_override'] = new_hz
            self.main_window.longitudinal_view.plot_areas()

        canvas = FigureCanvasQTAgg(sweep_fig)
        canvas.mpl_connect('button_press_event', on_sweep_click)

        win = QMainWindow(self.main_window)
        win.setWindowTitle('Breathing frequency sweep')
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(canvas)
        win.setCentralWidget(container)
        win.resize(1300, 500)
        win.show()
        self._breathing_sweep_win = win  # keep reference so Qt GC doesn't destroy it

    # ──────────────────────────────── mouse interaction ────

    def on_click(self, event):
        if self._toolbar_active():
            return
        if event.button is MouseButton.LEFT and event.inaxes:
            new_line = True
            set_dia = False
            set_sys = False
            set_slider_to = event.xdata

            if self.selected_line is not None:
                self.selected_line.set_linestyle(self.default_linestyle)
                self.selected_line = None

            if self.vertical_lines:
                distances = [abs(line.get_xdata()[0] - event.xdata) for line in self.vertical_lines]
                if min(distances) < len(self.frames) / 100:
                    self.selected_line = self.vertical_lines[np.argmin(distances)]
                    new_line = False
                    set_slider_to = self.selected_line.get_xdata()[0]

            if new_line:
                if self.current_phase == 'D':
                    color = self.main_window.diastole_color_plt
                    set_dia = True
                elif self.current_phase == 'S':
                    color = self.main_window.systole_color_plt
                    set_sys = True
                else:
                    color = self.default_line_color
                self.selected_line = self.ax.axvline(x=event.xdata, color=color, linestyle=self.default_linestyle)
                self.vertical_lines.append(self.selected_line)

            assert self.selected_line is not None
            self.selected_line.set_linestyle('dashed')
            self._draw()

            set_slider_to = round(set_slider_to - 1)
            self.main_window.display_slider.set_value(set_slider_to, reset_highlights=False)

            if set_slider_to in self.main_window.runtime_data.gated_frames_dia or set_dia:
                self.tmp_phase = 'D'
                toggle_diastolic_frame(self.main_window, False, drag=True)
            elif set_slider_to in self.main_window.runtime_data.gated_frames_sys or set_sys:
                self.tmp_phase = 'S'
                toggle_systolic_frame(self.main_window, False, drag=True)

    def on_release(self, event):
        if self._toolbar_active():
            return
        if event.button is MouseButton.LEFT and event.inaxes:
            if self.tmp_phase == 'D':
                self.main_window.diastolic_frame_box.setChecked(True)
                toggle_diastolic_frame(self.main_window, True, drag=True)
            elif self.tmp_phase == 'S':
                self.main_window.systolic_frame_box.setChecked(True)
                toggle_systolic_frame(self.main_window, True, drag=True)
        self.tmp_phase = None

    def on_motion(self, event):
        if self._toolbar_active():
            return
        if event.button is MouseButton.LEFT and self.selected_line:
            self.selected_line.set_xdata(np.array([event.xdata]))
            if event.xdata is not None:
                self.main_window.display_slider.set_value(round(event.xdata - 1), reset_highlights=False)
                self._draw()
            else:
                self.vertical_lines.remove(self.selected_line)
                self.selected_line = None
                self.tmp_phase = None
                self._draw()

    # ──────────────────────────────── display helpers ────

    def set_frame(self, frame):
        if not self._ready:
            return
        self.ax.set_autoscale_on(False)
        if self.frame_marker:
            try:
                self.frame_marker[0].remove()
            except NotImplementedError:
                pass
        self.frame_marker = self.ax.plot(frame + 1, self.ax.get_ylim()[0], 'yo', clip_on=False)
        self._draw()

    def draw_existing_lines(self, frames, color):
        if not self._ready:
            return
        # Only draw lines whose frame numbers fall within the current x range
        in_range = set(self.x)
        for frame in frames:
            if (frame + 1) in in_range:
                self.vertical_lines.append(self.ax.axvline(x=frame + 1, color=color, linestyle=self.default_linestyle))

    def remove_lines(self):
        for line in self.vertical_lines:
            try:
                line.remove()
            except NotImplementedError:
                pass
        self.vertical_lines = []
        self._draw()

    def redraw_phase_lines(self, frames_dia, color_dia, frames_sys, color_sys):
        self.remove_lines()
        self.draw_existing_lines(frames_dia, color_dia)
        self.draw_existing_lines(frames_sys, color_sys)
        self._draw()

    def update_color(self, color=None):
        color = color or self.default_line_color
        if self.selected_line is not None:
            self.selected_line.set_color(color)
            self._draw()

    def reset_highlights(self):
        if self.selected_line is not None:
            self.selected_line.set_linestyle(self.default_linestyle)
            self.selected_line = None
            self._draw()
