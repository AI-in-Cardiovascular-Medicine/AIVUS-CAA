import os
import time
import cv2
import numpy as np

from loguru import logger
from functools import partial
from PyQt5.QtGui import QKeySequence, QDesktopServices
from PyQt5.QtWidgets import QShortcut, QApplication
from PyQt5.QtCore import Qt, QUrl

from gui.popup_windows.frame_range_dialog import FrameRangeDialog
from gui.popup_windows.message_boxes import ErrorMessage, SuccessMessage
from gui.popup_windows.video_player import VideoPlayer
from gui.utils.contours_gui import new_contour, new_measure, new_eem
from input_output.metadata import MetadataWindow
from input_output.read_image import read_image
from input_output.contours_io import write_contours, save_gated_images
from segmentation.save_as_nifti import save_as_nifti
from segmentation.segment import segment
from report.report import report


from gui.popup_windows.results_plot import ResultsPlot


def init_shortcuts(main_window):
    # General
    QShortcut(QKeySequence('J'), main_window, partial(jiggle_frame, main_window))
    QShortcut(QKeySequence('Escape'), main_window, partial(stop_all, main_window))
    QShortcut(QKeySequence('Delete'), main_window, partial(delete_contour, main_window))
    QShortcut(QKeySequence('Ctrl+Z'), main_window, partial(undo_delete, main_window))
    # Gating
    QShortcut(QKeySequence('Alt+P'), main_window, partial(plot_results, main_window))
    QShortcut(QKeySequence('Alt+Delete'), main_window, partial(reset_phases, main_window))
    QShortcut(QKeySequence('Alt+S'), main_window, partial(switch_phases, main_window))
    # Traverse frames
    QShortcut(QKeySequence('W'), main_window, main_window.display_slider.next_gated_frame)
    QShortcut(QKeySequence(Qt.Key_Up), main_window, main_window.display_slider.next_gated_frame)
    QShortcut(QKeySequence('A'), main_window, main_window.display_slider.last_frame)
    QShortcut(QKeySequence(Qt.Key_Left), main_window, main_window.display_slider.last_frame)
    QShortcut(QKeySequence('S'), main_window, main_window.display_slider.last_gated_frame)
    QShortcut(QKeySequence(Qt.Key_Down), main_window, main_window.display_slider.last_gated_frame)
    QShortcut(QKeySequence('D'), main_window, main_window.display_slider.next_frame)
    QShortcut(QKeySequence(Qt.Key_Right), main_window, main_window.display_slider.next_frame)


def init_menu(main_window):
    file_menu = main_window.menu_bar.addMenu('File')
    open_action = file_menu.addAction('Open File', partial(read_image, main_window))
    open_action.setShortcut('Ctrl+O')
    file_menu.addSeparator()
    save_contours = file_menu.addAction('Save Contours', partial(write_contours, main_window))
    save_contours.setShortcut('Ctrl+S')
    nifti_menu = file_menu.addMenu('Save NIfTis')
    nifti_menu.addAction('Contoured Frames', partial(save_as_nifti, main_window, mode='contoured'))
    nifti_menu.addAction('Gated Frames', partial(save_as_nifti, main_window, mode='gated'))
    nifti_menu.addAction('All Frames', partial(save_as_nifti, main_window, mode='all'))
    save_report = file_menu.addAction('Save Report', partial(report, main_window))
    save_report.setShortcut('Ctrl+R')
    file_menu.addAction('Save Video Pullback', partial(save_video_pullback, main_window))
    file_menu.addAction('Save Gated Images', partial(save_gated_images, main_window, main_window.file_name))
    file_menu.addSeparator()
    exit_action = file_menu.addAction('Exit', main_window.close)
    exit_action.setShortcut('Ctrl+Q')

    edit_menu = main_window.menu_bar.addMenu('Edit')
    manual_contour_lumen = edit_menu.addAction('Manual Contour Lumen', partial(new_contour, main_window))
    manual_contour_lumen.setShortcut('E')
    manual_contour_eem = edit_menu.addAction('Manual Contour EEM', partial(new_eem, main_window))
    manual_contour_eem.setShortcut('Q')
    edit_menu.addAction('Remove Contours', partial(remove_contours, main_window))
    edit_menu.addSeparator()
    edit_menu.addAction('Reset Phases', partial(reset_phases, main_window))
    edit_menu.addSeparator()
    measure_1 = edit_menu.addAction('Measurement 1', partial(new_measure, main_window, index=0))
    measure_1.setShortcut('1')
    measure_2 = edit_menu.addAction('Measurement 2', partial(new_measure, main_window, index=1))
    measure_2.setShortcut('2')

    view_menu = main_window.menu_bar.addMenu('View')
    hide_contours_action = view_menu.addAction('Hide Contours', partial(hide_contours, main_window))
    hide_contours_action.setShortcut('H')
    hide_special_points_action = view_menu.addAction('Hide Special Points', partial(hide_special_points, main_window))
    hide_special_points_action.setShortcut('G')
    view_menu.addSeparator()
    reset_windowing_action = view_menu.addAction('Reset Windowing', partial(reset_windowing, main_window))
    reset_windowing_action.setShortcut('R')
    toggle_color_action = view_menu.addAction('Toggle Color', partial(toggle_color, main_window))
    toggle_color_action.setShortcut('C')
    view_menu.addSeparator()
    filter_1 = view_menu.addAction('Apply Median Blur', partial(toggle_filter, main_window, index=0))
    filter_1.setShortcut('3')
    filter_2 = view_menu.addAction('Apply Gaussian Blur', partial(toggle_filter, main_window, index=1))
    filter_2.setShortcut('4')
    filter_3 = view_menu.addAction('Apply Bilateral Filter', partial(toggle_filter, main_window, index=2))
    filter_3.setShortcut('5')

    run_menu = main_window.menu_bar.addMenu('Run')
    run_menu.addAction('Extract Diastolic and Systolic Frames', main_window.contour_based_gating)
    run_menu.addAction('Automatic Segmentation', partial(segment, main_window))

    metadata_menu = main_window.menu_bar.addMenu('Metadata')
    metadata_menu.addAction('Show Metadata', partial(show_metadata, main_window))

    help_menu = main_window.menu_bar.addMenu('Help')
    help_menu.addAction('GitHub Page', partial(open_url, main_window, description='github'))
    help_menu.addAction('Keyboard Shortcuts', partial(open_url, main_window, description='keyboard_shortcuts'))
    help_menu.addSeparator()
    help_menu.addAction('About', partial(open_url, main_window))


def is_gating_display_active(main_window):
    """
    Checks if an image is displayed in the gating display box.

    Parameters:
        main_window: The main window containing the gating display.

    Returns:
        bool: True if the gating display contains an image, False otherwise.
    """
    return (
        main_window.gating_display is not None
        and main_window.gating_display.fig.axes  # Check if axes exist
        and any(ax.has_data() for ax in main_window.gating_display.fig.axes)  # Check if any axis has data
    )


def remove_contours(main_window):
    if main_window.image_displayed:
        dialog = FrameRangeDialog(main_window)
        if (dialog.exec_()):
            main_window.status_bar.showMessage('Removing contours...')
            lower_limit, upper_limit = dialog.getInputs()
            for frame in range(lower_limit, upper_limit):
                main_window.data['lumen'][0][frame] = []
                main_window.data['lumen'][1][frame] = []
                main_window.data['eem'][0][frame] = []
                main_window.data['eem'][1][frame] = []
            main_window.longitudinal_view.remove_contours(lower_limit, upper_limit)
            main_window.display.update_display()
            main_window.status_bar.showMessage(main_window.waiting_status)


def reset_phases(main_window):
    if main_window.image_displayed:
        dialog = FrameRangeDialog(main_window)
        if (dialog.exec_()):
            main_window.status_bar.showMessage('Resetting phases...')
            lower_limit, upper_limit = dialog.getInputs()
            for frame in range(lower_limit, upper_limit):
                if main_window.data['phases'][frame] == 'D':
                    main_window.gated_frames_dia.remove(frame)
                    main_window.diastolic_frame_box.setChecked(False)
                elif main_window.data['phases'][frame] == 'S':
                    main_window.gated_frames_sys.remove(frame)
                    main_window.systolic_frame_box.setChecked(False)
                main_window.data['phases'][frame] = '-'
            main_window.gated_frames = main_window.gated_frames_dia + main_window.gated_frames_sys
            main_window.gated_frames.sort()
            main_window.gated_frames_dia.sort()
            main_window.gated_frames_sys.sort()
            main_window.status_bar.showMessage(main_window.waiting_status)

            main_window.contour_based_gating.remove_lines()
            main_window.contour_based_gating.draw_existing_lines(
                main_window.gated_frames_dia, main_window.diastole_color_plt
            )
            main_window.contour_based_gating.draw_existing_lines(
                main_window.gated_frames_sys, main_window.systole_color_plt
            )  # somehow only updates after first user input

            main_window.display.update_display()


def switch_phases(main_window):
    # Check if gating display is active; if not, show a message and return
    if not is_gating_display_active(main_window):
        ErrorMessage(main_window, 'Please extract diastolic and systolic frames first.')
        return

    if main_window.image_displayed:
        dialog = FrameRangeDialog(main_window)
        if (dialog.exec_()):
            main_window.status_bar.showMessage('Switching phases...')
            lower_limit, upper_limit = dialog.getInputs()
            for frame in range(lower_limit, upper_limit):
                if main_window.data['phases'][frame] == 'D':
                    main_window.data['phases'][frame] = 'S'
                    main_window.gated_frames_dia.remove(frame)
                    main_window.gated_frames_sys.append(frame)
                    main_window.diastolic_frame_box.setChecked(False)
                    main_window.systolic_frame_box.setChecked(True)
                elif main_window.data['phases'][frame] == 'S':
                    main_window.data['phases'][frame] = 'D'
                    main_window.gated_frames_sys.remove(frame)
                    main_window.gated_frames_dia.append(frame)
                    main_window.diastolic_frame_box.setChecked(True)
                    main_window.systolic_frame_box.setChecked(False)

            main_window.gated_frames = main_window.gated_frames_dia + main_window.gated_frames_sys

        # order all gated frames again, important otherwise slider will jump around
        main_window.gated_frames.sort()
        main_window.gated_frames_dia.sort()
        main_window.gated_frames_sys.sort()
        main_window.status_bar.showMessage(main_window.waiting_status)

        # Call draw_existing_lines on the ContourBasedGating instance, but first remove all existing lines to live update plot
        main_window.contour_based_gating.remove_lines()
        main_window.contour_based_gating.draw_existing_lines(
            main_window.gated_frames_dia, main_window.diastole_color_plt
        )
        main_window.contour_based_gating.draw_existing_lines(
            main_window.gated_frames_sys, main_window.systole_color_plt
        )

        main_window.display.update_display()


def show_metadata(main_window):
    if main_window.image_displayed:
        metadata_window = MetadataWindow(main_window)
        metadata_window.show()


def open_url(main_window, description=None):
    if description == 'github':
        url = 'https://github.com/cardionaut/AAOCASeg'
    elif description == 'keyboard_shortcuts':
        url = 'https://github.com/cardionaut/AAOCASeg?tab=readme-ov-file#keyboard-shortcuts'
    else:
        video_player = VideoPlayer(main_window)
        video_player.play('media/about.mp4')
        video_player.move(main_window.x() + main_window.width() // 2, main_window.y() + main_window.height() // 2)
        return
    if not QDesktopServices.openUrl(QUrl(url)):
        ErrorMessage(main_window, 'Could not open the browser. Please visit\n' + url)


def hide_contours(main_window):
    if main_window.image_displayed:
        main_window.hide_contours_box.setChecked(not main_window.hide_contours_box.isChecked())


def hide_special_points(main_window):
    if main_window.image_displayed:
        main_window.hide_special_points_box.setChecked(not main_window.hide_special_points_box.isChecked())


def jiggle_frame(main_window):
    if main_window.image_displayed:
        current_frame = main_window.display_slider.value()
        main_window.display_slider.set_value(current_frame + 1)
        QApplication.processEvents()
        time.sleep(0.1)
        main_window.display_slider.set_value(current_frame)
        QApplication.processEvents()
        time.sleep(0.1)
        main_window.display_slider.set_value(current_frame - 1)
        QApplication.processEvents()
        time.sleep(0.1)
        main_window.display_slider.set_value(current_frame)
        QApplication.processEvents()


def toggle_filter(main_window, index):
    if main_window.image_displayed:
        if main_window.filter == index:
            main_window.filter = None
        else:
            main_window.filter = index
        main_window.display.display_image(update_image=True)


def stop_all(main_window):
    main_window.display.stop_contour()
    main_window.display.measure_index = None


def delete_contour(main_window):
    if main_window.image_displayed:
        main_window.tmp_lumen_x = main_window.data['lumen'][0][main_window.display.frame]  # for Ctrl+Z
        main_window.tmp_lumen_y = main_window.data['lumen'][1][main_window.display.frame]
        main_window.tmp_eem_x = main_window.data['eem'][0][main_window.display.frame]
        main_window.tmp_eem_y = main_window.data['eem'][1][main_window.display.frame]
        main_window.data['lumen'][0][main_window.display.frame] = []
        main_window.data['lumen'][1][main_window.display.frame] = []
        main_window.data['eem'][0][main_window.display.frame] = []
        main_window.data['eem'][1][main_window.display.frame] = []
        main_window.display.display_image(update_contours=True)


def undo_delete(main_window):
    if main_window.image_displayed and main_window.tmp_lumen_x:
        main_window.data['lumen'][0][main_window.display.frame] = main_window.tmp_lumen_x
        main_window.data['lumen'][1][main_window.display.frame] = main_window.tmp_lumen_y
        main_window.data['eem'][0][main_window.display.frame] = main_window.tmp_eem_x
        main_window.data['eem'][1][main_window.display.frame] = main_window.tmp_eem_y
        main_window.tmp_lumen_x = []
        main_window.tmp_lumen_y = []
        main_window.tmp_eem_x = []
        main_window.tmp_eem_y = []
    main_window.display.stop_contour()


def reset_windowing(main_window):
    if main_window.image_displayed:
        main_window.display.window_level = main_window.display.initial_window_level
        main_window.display.window_width = main_window.display.initial_window_width
        main_window.display.display_image(update_image=True)


def toggle_color(main_window):
    if main_window.image_displayed:
        main_window.colormap_enabled = not main_window.colormap_enabled
        main_window.display.display_image(update_image=True)


def plot_results(main_window):
    if main_window.image_displayed:
        report_data = report(main_window, suppress_messages=True)
        if report_data is None:
            logger.error('No report data available to plot')
            return
        results_plot = ResultsPlot(main_window, report_data)
        results_plot.show()


def save_video_pullback(main_window):
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot save video pullback before reading the image.')
        return
    main_window.status_bar.showMessage('Saving video pullback...')
    image_stack = main_window.images
    size = (image_stack[0].shape[1], image_stack[0].shape[0])
    fps = main_window.metadata['frame_rate']
    duration = len(image_stack) // fps
    out_path = os.path.splitext(main_window.file_name)[0] + '_pullback.mp4'
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), False)
    for frame in range(fps * duration):
        out.write(image_stack[frame, :, :])
    out.release()
    SuccessMessage(main_window, f'Saving video')
    main_window.status_bar.showMessage(main_window.waiting_status)
