import os
import time
import cv2
import numpy as np

from loguru import logger
from functools import partial
from PyQt6.QtGui import QKeySequence, QDesktopServices, QShortcut
from PyQt6.QtWidgets import QApplication, QProgressDialog
from PyQt6.QtCore import Qt, QUrl

from pages.intravascular.popup_windows.frame_range_dialog import FrameRangeDialog
from pages.intravascular.popup_windows.message_boxes import ErrorMessage, SuccessMessage
from pages.intravascular.popup_windows.video_player import VideoPlayer
from pages.intravascular.utils.contours_gui import new_contour, new_contour_append, new_measure, new_angle, set_tool
from input_output.input.metadata import CctaMetadataWindow, MetadataWindow
from input_output.input.image import read_image, read_nifti_mask
from input_output.output.contours import write_contours
from input_output.output.other_fmt import save_gated_images
from input_output.output.imgs_masks import save_as_nifti
from input_output.output.reports import report


from pages.intravascular.popup_windows.results_plot import ResultsPlot
from domain.all_types import ContourType, SegmentationTool


def _sync_contour_combo(main_window, contour_type):
    lh = getattr(main_window, 'left_half', None)
    if lh is not None:
        lh.set_active_contour_type_ui(contour_type)


def _new_contour_synced(main_window, contour_type):
    new_contour(main_window, contour_type)
    _sync_contour_combo(main_window, contour_type)


def _new_contour_append_synced(main_window, contour_type):
    new_contour_append(main_window, contour_type)
    _sync_contour_combo(main_window, contour_type)


def init_ccta_shortcuts(ccta_page):
    _widget_children_shortcut('R', ccta_page, ccta_page.reset_windowing)
    _widget_children_shortcut('F', ccta_page, ccta_page.reset_zoom)
    _widget_children_shortcut('Escape', ccta_page, ccta_page.reset_to_neutral)


def _widget_children_shortcut(key, parent, slot):
    """Create a QShortcut scoped to parent and its children only."""
    sc = QShortcut(QKeySequence(key), parent, slot)
    sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
    return sc


def init_shortcuts(main_window):
    # scoped so R/F don't conflict with CCTA shortcuts
    _widget_children_shortcut('R', main_window, partial(reset_windowing, main_window))
    _widget_children_shortcut('F', main_window, partial(reset_zoom, main_window))
    # General
    QShortcut(QKeySequence('J'), main_window, partial(jiggle_frame, main_window))
    QShortcut(QKeySequence('Escape'), main_window, partial(stop_all, main_window))
    QShortcut(QKeySequence('Delete'), main_window, partial(delete_contour, main_window))
    QShortcut(QKeySequence('Ctrl+Z'), main_window, partial(undo_delete, main_window))
    QShortcut(QKeySequence('Shift+A'), main_window, partial(copy_contour_from_left, main_window))
    QShortcut(QKeySequence('Shift+D'), main_window, partial(copy_contour_from_right, main_window))
    QShortcut(QKeySequence('Shift+W'), main_window, partial(copy_contour_from_next_gated, main_window))
    QShortcut(QKeySequence('Shift+S'), main_window, partial(copy_contour_from_prev_gated, main_window))
    # Gating
    QShortcut(QKeySequence('Alt+P'), main_window, partial(plot_results, main_window))
    QShortcut(QKeySequence('Alt+Delete'), main_window, partial(reset_phases, main_window))
    QShortcut(QKeySequence('Alt+S'), main_window, partial(switch_phases, main_window))
    # Traverse frames
    QShortcut(QKeySequence('W'), main_window, lambda: main_window.display_slider.next_gated_frame())
    QShortcut(QKeySequence(Qt.Key.Key_Up), main_window, lambda: main_window.display_slider.next_gated_frame())
    QShortcut(QKeySequence('A'), main_window, lambda: main_window.display_slider.last_frame())
    QShortcut(QKeySequence(Qt.Key.Key_Left), main_window, lambda: main_window.display_slider.last_frame())
    QShortcut(QKeySequence('S'), main_window, lambda: main_window.display_slider.last_gated_frame())
    QShortcut(QKeySequence(Qt.Key.Key_Down), main_window, lambda: main_window.display_slider.last_gated_frame())
    QShortcut(QKeySequence('D'), main_window, lambda: main_window.display_slider.next_frame())
    QShortcut(QKeySequence(Qt.Key.Key_Right), main_window, lambda: main_window.display_slider.next_frame())


def _save_current_page(main_window, ccta_page) -> None:
    from gui.active_page import ActivePage

    master = main_window.window()
    if hasattr(master, 'active_page') and master.active_page == ActivePage.CCTA:
        ccta_page.save_mask()
    else:
        write_contours(main_window)


def init_menu(main_window, ccta_page):
    file_menu = main_window.menu_bar.addMenu('File')
    open_action = file_menu.addAction('Open Intravascular File', partial(read_image, main_window))
    open_action.setShortcut('Ctrl+O')
    file_menu.addAction('Open Intravascular Mask', partial(read_nifti_mask, main_window))
    file_menu.addSeparator()
    open_folder_action = file_menu.addAction('Open CCTA Folder/File', ccta_page.open_folder)
    open_folder_action.setShortcut('Ctrl+Shift+O')
    file_menu.addAction('Open CCTA Mask', ccta_page.open_mask)
    file_menu.addSeparator()
    save_contours = file_menu.addAction('Save', partial(_save_current_page, main_window, ccta_page))
    save_contours.setShortcut('Ctrl+S')
    nifti_menu = file_menu.addMenu('Save NIfTis')
    nifti_menu.addAction('Contoured Frames', partial(save_as_nifti, main_window, mode='contoured'))
    nifti_menu.addAction('Gated Frames', partial(save_as_nifti, main_window, mode='gated'))
    nifti_menu.addAction('All Frames', partial(save_as_nifti, main_window, mode='all'))
    save_report = file_menu.addAction('Save Report', partial(report, main_window))
    save_report.setShortcut('Ctrl+R')
    file_menu.addAction('Save Video Pullback', partial(save_video_pullback, main_window))
    file_menu.addAction('Save Gated Images', partial(save_gated_images, main_window))
    file_menu.addSeparator()
    exit_action = file_menu.addAction('Exit', main_window.close)
    exit_action.setShortcut('Ctrl+Q')

    edit_menu = main_window.menu_bar.addMenu('Edit')
    manual_lumen_contour = edit_menu.addAction(
        'Manual Lumen Contour', partial(_new_contour_synced, main_window, ContourType.LUMEN)
    )
    manual_lumen_contour.setShortcut('E')
    manual_eem_contour = edit_menu.addAction(
        'Manual EEM Contour', partial(_new_contour_synced, main_window, ContourType.EEM)
    )
    manual_eem_contour.setShortcut('Q')
    spawn_eem_action = edit_menu.addAction('Spawn EEM from Lumen', partial(spawn_eem_from_lumen, main_window))
    spawn_eem_action.setShortcut('Shift+Q')
    manual_calc_contour = edit_menu.addAction(
        'Manual Calcium Contour', partial(_new_contour_synced, main_window, ContourType.CALCIUM)
    )
    manual_calc_contour.setShortcut('7')
    manual_branch_contour = edit_menu.addAction(
        'Manual Branch Contour', partial(_new_contour_synced, main_window, ContourType.BRANCH)
    )
    manual_branch_contour.setShortcut('8')
    manual_lipid_contour = edit_menu.addAction(
        'Manual Lipid Contour', partial(_new_contour_synced, main_window, ContourType.LIPID)
    )
    manual_lipid_contour.setShortcut('9')
    manual_macroph_contour = edit_menu.addAction(
        'Manual Macrophage Contour', partial(_new_contour_synced, main_window, ContourType.MACROPHAGE)
    )
    manual_macroph_contour.setShortcut('0')
    edit_menu.addSeparator()
    add_calc_contour = edit_menu.addAction(
        'Add Calcium Contour', partial(_new_contour_append_synced, main_window, ContourType.CALCIUM)
    )
    add_calc_contour.setShortcut('Ctrl+7')
    add_branch_contour = edit_menu.addAction(
        'Add Branch Contour', partial(_new_contour_append_synced, main_window, ContourType.BRANCH)
    )
    add_branch_contour.setShortcut('Ctrl+8')
    add_lipid_contour = edit_menu.addAction(
        'Add Lipid Contour', partial(_new_contour_append_synced, main_window, ContourType.LIPID)
    )
    add_lipid_contour.setShortcut('Ctrl+9')
    add_macroph_contour = edit_menu.addAction(
        'Add Macrophage Contour', partial(_new_contour_append_synced, main_window, ContourType.MACROPHAGE)
    )
    add_macroph_contour.setShortcut('Ctrl+0')
    edit_menu.addAction('Remove Contours', partial(remove_contours, main_window))
    edit_menu.addSeparator()
    edit_menu.addAction('Reset Phases', partial(reset_phases, main_window))
    edit_menu.addSeparator()
    measure_1 = edit_menu.addAction('Measurement 1', partial(new_measure, main_window, index=0))
    measure_1.setShortcut('1')
    measure_2 = edit_menu.addAction('Measurement 2', partial(new_measure, main_window, index=1))
    measure_2.setShortcut('2')
    angle_wire = edit_menu.addAction('Angle Wire Shadow', partial(new_angle, main_window, ContourType.WIRE))
    angle_wire.setShortcut('3')
    closed_spline = edit_menu.addAction('Closed Spline', partial(set_tool, main_window, SegmentationTool.CLOSED_SPLINE))
    closed_spline.setShortcut('4')
    open_spline = edit_menu.addAction('Open Spline', partial(set_tool, main_window, SegmentationTool.OPEN_SPLINE))
    open_spline.setShortcut('5')
    brush = edit_menu.addAction('Brush', partial(set_tool, main_window, SegmentationTool.BRUSH))
    brush.setShortcut('6')

    view_menu = main_window.menu_bar.addMenu('View')
    hide_contours_action = view_menu.addAction('Hide Contours', partial(hide_contours, main_window))
    hide_contours_action.setShortcut('H')
    hide_special_points_action = view_menu.addAction('Hide Measurements', partial(hide_special_points, main_window))
    hide_special_points_action.setShortcut('G')
    view_menu.addSeparator()
    view_menu.addAction('Reset Windowing', partial(reset_windowing, main_window))
    view_menu.addAction('Reset Zoom', partial(reset_zoom, main_window))
    toggle_color_action = view_menu.addAction('Toggle Color', partial(toggle_color, main_window))
    toggle_color_action.setShortcut('C')
    view_menu.addSeparator()

    run_menu = main_window.menu_bar.addMenu('Run')
    run_menu.addAction('Extract Diastolic and Systolic Frames', main_window.gating_plot)
    # run_menu.addAction('Automatic Segmentation', partial(segment, main_window))

    metadata_menu = main_window.menu_bar.addMenu('Metadata')
    metadata_menu.addAction('Show Metadata', partial(show_metadata, main_window))

    help_menu = main_window.menu_bar.addMenu('Help')
    help_menu.addAction('GitHub Page', partial(open_url, main_window, description='github'))
    help_menu.addAction('Documentation', partial(open_url, main_window, description='docs'))
    help_menu.addAction('Keyboard Shortcuts', partial(open_url, main_window, description='keyboard_shortcuts'))
    help_menu.addAction('Report a Problem', partial(open_url, main_window, description='issue'))
    help_menu.addAction('Request a Feature', partial(open_url, main_window, description='feature'))
    help_menu.addSeparator()
    help_menu.addAction('About', partial(open_url, main_window))


def _copy_contour_from_frame(main_window, source_frame: int) -> None:
    """Replace the active contour on the current frame with a copy from source_frame."""
    display = main_window.display
    current_frame = display.frame
    key = display.contour_key()
    ci = display.active_contour_index

    fd_src = main_window.runtime_data.frame_data_dct.get(source_frame)
    fd_dst = main_window.runtime_data.frame_data_dct.get(current_frame)
    if fd_src is None or fd_dst is None:
        return

    src_obj = getattr(fd_src, key, None)
    dst_obj = getattr(fd_dst, key, None)
    if src_obj is None or dst_obj is None:
        return
    if not hasattr(src_obj, 'contours') or not src_obj.contours or ci >= len(src_obj.contours):
        return

    src_contour = src_obj.contours[ci]
    xs = list(src_contour[0]) if src_contour else []
    ys = list(src_contour[1]) if len(src_contour) > 1 else []
    src_start = list(src_obj.start_coords[ci]) if ci < len(src_obj.start_coords) else []
    src_end = list(src_obj.end_coords[ci]) if ci < len(src_obj.end_coords) else []
    src_closed = src_obj.closed[ci] if ci < len(src_obj.closed) else True

    while len(dst_obj.contours) <= ci:
        dst_obj.contours.append([[], []])
    while len(dst_obj.start_coords) <= ci:
        dst_obj.start_coords.append([])
    while len(dst_obj.end_coords) <= ci:
        dst_obj.end_coords.append([])
    while len(dst_obj.closed) <= ci:
        dst_obj.closed.append(True)

    dst_obj.contours[ci] = [xs, ys]
    dst_obj.start_coords[ci] = src_start
    dst_obj.end_coords[ci] = src_end
    dst_obj.closed[ci] = src_closed

    display.update_display()
    try:
        main_window.longitudinal_view.plot_areas()
    except Exception as e:
        logger.debug(f"Could not update longitudinal view after contour copy: {e}")


def copy_contour_from_left(main_window):
    if not main_window.image_displayed:
        return
    frame = main_window.display.frame
    if frame <= 0:
        return
    _copy_contour_from_frame(main_window, frame - 1)


def copy_contour_from_right(main_window):
    if not main_window.image_displayed:
        return
    display = main_window.display
    frame = display.frame
    n_frames = display.images.shape[0] if display.images is not None else 0
    if frame >= n_frames - 1:
        return
    _copy_contour_from_frame(main_window, frame + 1)


def copy_contour_from_next_gated(main_window):
    if not main_window.image_displayed:
        return
    frame = main_window.display.frame
    gated = sorted(main_window.runtime_data.gated_frames)
    if frame not in gated:
        return
    later = [f for f in gated if f > frame]
    if not later:
        return
    _copy_contour_from_frame(main_window, later[0])


def copy_contour_from_prev_gated(main_window):
    if not main_window.image_displayed:
        return
    frame = main_window.display.frame
    gated = sorted(main_window.runtime_data.gated_frames)
    if frame not in gated:
        return
    earlier = [f for f in gated if f < frame]
    if not earlier:
        return
    _copy_contour_from_frame(main_window, earlier[-1])


def spawn_eem_from_lumen(main_window):
    """Ctrl+E: create an EEM contour by expanding the lumen knot points 20% radially from their centroid."""
    if not main_window.image_displayed:
        return
    frame = main_window.display.frame
    fd = main_window.runtime_data.frame_data_dct.get(frame)
    if fd is None:
        return

    eem_obj = getattr(fd, ContourType.EEM.value, None)
    if eem_obj is None:
        return
    if eem_obj.contours and eem_obj.contours[0] and eem_obj.contours[0][0]:
        return  # EEM already exists on this frame

    lumen_obj = getattr(fd, ContourType.LUMEN.value, None)
    if lumen_obj is None or not lumen_obj.contours or not lumen_obj.contours[0] or not lumen_obj.contours[0][0]:
        return

    xs = list(lumen_obj.contours[0][0])
    ys = list(lumen_obj.contours[0][1]) if len(lumen_obj.contours[0]) > 1 else []
    if len(xs) < 2 or len(xs) != len(ys):
        return

    cx = sum(xs) / len(xs)
    cy = sum(ys) / len(ys)
    scale = 1.1

    eem_obj.contours = [[[cx + (x - cx) * scale for x in xs], [cy + (y - cy) * scale for y in ys]]]
    eem_obj.closed = [True]
    eem_obj.start_coords = [[]]
    eem_obj.end_coords = [[]]

    main_window.display.active_contour_type = ContourType.EEM
    main_window.display.active_contour_index = 0
    _sync_contour_combo(main_window, ContourType.EEM)
    main_window.display.update_display()


def remove_contours(main_window):
    if main_window.image_displayed:
        dialog = FrameRangeDialog(main_window)
        if dialog.exec():
            main_window.status_bar.showMessage('Removing contours...')
            lower_limit, upper_limit = dialog.getInputs()
            key = main_window.display.contour_key()
            for frame in range(lower_limit, upper_limit):
                fd = main_window.runtime_data.frame_data_dct.get(frame)
                if fd:
                    contour_obj = getattr(fd, key, None)
                    if contour_obj:
                        contour_obj.contours = []
            main_window.longitudinal_view.remove_contours(lower_limit, upper_limit)
            main_window.display.update_display()
            main_window.status_bar.showMessage(main_window.waiting_status)


def reset_phases(main_window):
    if main_window.image_displayed:
        dialog = FrameRangeDialog(main_window)
        if dialog.exec():
            main_window.status_bar.showMessage('Resetting phases...')
            lower_limit, upper_limit = dialog.getInputs()
            for frame in range(lower_limit, upper_limit):
                fd = main_window.runtime_data.frame_data_dct.get(frame)
                if fd is None:
                    continue
                if fd.phase == 'D':
                    main_window.runtime_data.gated_frames_dia.remove(frame)
                    main_window.diastolic_frame_box.setChecked(False)
                elif fd.phase == 'S':
                    main_window.runtime_data.gated_frames_sys.remove(frame)
                    main_window.systolic_frame_box.setChecked(False)
                elif fd.phase == 'T':
                    try:
                        main_window.runtime_data.tagged_frames.remove(frame)
                    except ValueError:
                        pass
                fd.phase = '-'
            if main_window.runtime_data.metadata.get('modality') == 'OCT':
                main_window.runtime_data.gated_frames = main_window.runtime_data.tagged_frames
            else:
                main_window.runtime_data.gated_frames = (
                    main_window.runtime_data.gated_frames_dia + main_window.runtime_data.gated_frames_sys
                )
                main_window.runtime_data.gated_frames.sort()
            main_window.runtime_data.gated_frames_dia.sort()
            main_window.runtime_data.gated_frames_sys.sort()
            main_window.status_bar.showMessage(main_window.waiting_status)

            _redraw_phase_views(main_window)


def _redraw_phase_views(main_window):
    """Refresh display, longitudinal view and gating plot after any phase change."""
    main_window.display.update_display()
    main_window.longitudinal_view.plot_areas()
    main_window.gating_plot.redraw_phase_lines(
        main_window.runtime_data.gated_frames_dia,
        main_window.diastole_color_plt,
        main_window.runtime_data.gated_frames_sys,
        main_window.systole_color_plt,
    )


def switch_phases(main_window):
    if main_window.runtime_data.metadata.get('modality') == 'OCT':
        return
    if not _is_gating_display_active(main_window):
        ErrorMessage(main_window, 'Please extract diastolic and systolic frames first.')
        return

    if main_window.image_displayed:
        dialog = FrameRangeDialog(main_window)
        if dialog.exec():
            main_window.status_bar.showMessage('Switching phases...')
            lower_limit, upper_limit = dialog.getInputs()
            for frame in range(lower_limit, upper_limit):
                fd = main_window.runtime_data.frame_data_dct.get(frame)
                if fd is None:
                    continue
                if fd.phase == 'D':
                    fd.phase = 'S'
                    main_window.runtime_data.gated_frames_dia.remove(frame)
                    main_window.runtime_data.gated_frames_sys.append(frame)
                    main_window.diastolic_frame_box.setChecked(False)
                    main_window.systolic_frame_box.setChecked(True)
                elif fd.phase == 'S':
                    fd.phase = 'D'
                    main_window.runtime_data.gated_frames_sys.remove(frame)
                    main_window.runtime_data.gated_frames_dia.append(frame)
                    main_window.diastolic_frame_box.setChecked(True)
                    main_window.systolic_frame_box.setChecked(False)

            main_window.runtime_data.gated_frames = (
                main_window.runtime_data.gated_frames_dia + main_window.runtime_data.gated_frames_sys
            )

        # order all gated frames again, important otherwise slider will jump around
        main_window.runtime_data.gated_frames.sort()
        main_window.runtime_data.gated_frames_dia.sort()
        main_window.runtime_data.gated_frames_sys.sort()
        main_window.status_bar.showMessage(main_window.waiting_status)

        _redraw_phase_views(main_window)


def _is_gating_display_active(main_window):
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


def show_metadata(main_window):
    from gui.active_page import ActivePage

    master = main_window.window()  # Master, not IntravascularPage
    if hasattr(master, 'active_page') and master.active_page == ActivePage.CCTA:
        win = CctaMetadataWindow(master, master.ccta_metadata)
        win.show()
    elif main_window.image_displayed:
        MetadataWindow(main_window).show()


def open_url(main_window, description=None):
    if description == 'github':
        url = 'https://github.com/yungselm/AIVUS-OCT'
    elif description == 'docs':
        url = 'https://aivus-caa.readthedocs.io/en/latest'
    elif description == 'keyboard_shortcuts':
        url = 'https://aivus-caa.readthedocs.io/en/latest/contents/usage.html'
    elif description == 'issue':
        url = 'https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/issues/new?template=bug_report.md'
    elif description == 'feature':
        url = 'https://github.com/AI-in-Cardiovascular-Medicine/AIVUS-CAA/issues/new?template=feature_request.md'
    else:
        video_player = VideoPlayer(main_window)
        video_player.play('../media/about.mp4')
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


def stop_all(main_window):
    main_window.display.disable_brush()
    main_window.display._interrupt_drawing_mode()
    main_window.display.active_contour_type = ContourType.LUMEN
    main_window.display.active_segmentation_tool = SegmentationTool.CLOSED_SPLINE
    main_window.left_half.brush_btn.setChecked(False)
    main_window.left_half.closed_spline_btn.setChecked(True)


def delete_contour(main_window):
    if main_window.image_displayed:
        key = main_window.display.contour_key()
        c_idx = main_window.display.active_contour_index

        if not hasattr(main_window, 'tmp_contours'):
            main_window.runtime_data.tmp_contours = {}

        frame = main_window.display.frame
        fd = main_window.runtime_data.frame_data_dct.get(frame)
        if fd:
            contour_obj = getattr(fd, key, None)
            if contour_obj and contour_obj.contours and c_idx < len(contour_obj.contours):
                c = contour_obj.contours[c_idx]
                xlist = list(c[0]) if c and c[0] else []
                ylist = list(c[1]) if c and len(c) > 1 else []
                start = contour_obj.start_coords[c_idx] if len(contour_obj.start_coords) > c_idx else []
                end = contour_obj.end_coords[c_idx] if len(contour_obj.end_coords) > c_idx else []
                closed = contour_obj.closed[c_idx] if len(contour_obj.closed) > c_idx else True
            else:
                xlist, ylist, start, end, closed = [], [], [], [], True

            main_window.runtime_data.tmp_contours[key] = (c_idx, xlist, ylist, start, end, closed)

            if contour_obj and c_idx < len(contour_obj.contours):
                del contour_obj.contours[c_idx]
                if c_idx < len(contour_obj.start_coords):
                    del contour_obj.start_coords[c_idx]
                if c_idx < len(contour_obj.end_coords):
                    del contour_obj.end_coords[c_idx]
                if c_idx < len(contour_obj.closed):
                    del contour_obj.closed[c_idx]

                # Update finalized_splines
                lst = main_window.display.finalized_splines.get(key)
                if lst and c_idx < len(lst):
                    del lst[c_idx]

                # Clamp active index
                remaining = len(contour_obj.contours)
                if remaining > 0:
                    main_window.display.active_contour_index = min(c_idx, remaining - 1)
                else:
                    main_window.display.active_contour_index = 0

        main_window.display.display_image(update_contours=True)


# TODO: Not working as intended
def undo_delete(main_window):
    if main_window.image_displayed:
        key = main_window.display.contour_key()
        if hasattr(main_window, 'tmp_contours') and key in main_window.runtime_data.tmp_contours:
            saved = main_window.runtime_data.tmp_contours.pop(key)
            ci, xlist, ylist, start, end, closed = saved
            frame = main_window.display.frame
            fd = main_window.runtime_data.frame_data_dct.get(frame)
            if fd:
                contour_obj = getattr(fd, key, None)
                if contour_obj is not None:
                    contour_obj.contours.insert(ci, [xlist, ylist])
                    contour_obj.start_coords.insert(ci, start)
                    contour_obj.end_coords.insert(ci, end)
                    contour_obj.closed.insert(ci, closed)
                    main_window.display.active_contour_index = ci

            main_window.display.update_display()


def reset_zoom(main_window):
    if main_window.image_displayed:
        main_window.display.resetTransform()


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
        main_window.results_plot = ResultsPlot(main_window, report_data)
        main_window.results_plot.show()


def save_video_pullback(main_window):
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot save video pullback before reading the image.')
        return
    main_window.status_bar.showMessage('Saving video pullback...')

    images_rgb = main_window.runtime_data.images_rgb
    images_gray = main_window.runtime_data.images
    use_color = images_rgb is not None
    image_stack = images_rgb if use_color else images_gray
    if image_stack is None:
        ErrorMessage(main_window, 'No image data available.')
        return

    n_frames = len(image_stack)
    fps = int(main_window.runtime_data.metadata.get('frame_rate', 30))
    if fps <= 0:
        fps = 30

    H, W = image_stack[0].shape[:2]
    out_path = os.path.splitext(main_window.file_name)[0] + '_pullback.mp4'
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W, H), use_color)  # type: ignore[attr-defined]
    wl = main_window.display.window_level
    ww = main_window.display.window_width
    lo, hi = wl - ww / 2, wl + ww / 2

    progress = QProgressDialog('Saving video pullback...', 'Cancel', 0, n_frames, main_window)
    progress.setWindowTitle('Saving Video')
    progress.setMinimumDuration(0)
    progress.setModal(True)
    progress.setValue(0)
    QApplication.processEvents()

    for i in range(n_frames):
        progress.setValue(i)
        QApplication.processEvents()
        if progress.wasCanceled():
            break
        frame_data = image_stack[i]
        if use_color:
            out.write(cv2.cvtColor(frame_data, cv2.COLOR_RGB2BGR))
        else:
            windowed = ((np.clip(frame_data.astype(float), lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)
            out.write(windowed)

    out.release()
    progress.close()
    SuccessMessage(main_window, 'Saving video')
    main_window.status_bar.showMessage(main_window.waiting_status)
