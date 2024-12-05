import os

import pydicom as dcm
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from PyQt5.QtWidgets import QFileDialog

from gui.popup_windows.message_boxes import ErrorMessage
from input_output.metadata import parse_dicom
from input_output.contours_io import read_contours


def read_image(main_window):
    """
    Reads DICOM or NIfTi images.

    Reads the DICOM/NIfTi images and metadata. Places metatdata in a table.
    Images are displayed in the graphics scene.
    """
    main_window.status_bar.showMessage('Reading image file...')
    options = QFileDialog.Options()
    options = QFileDialog.DontUseNativeDialog
    file_name, _ = QFileDialog.getOpenFileName(
        main_window, 'QFileDialog.getOpenFileName()', '..', 'All files (*)', options=options
    )

    if file_name:
        main_window.gating_display.fig.clear()
        plt.draw()
        try:  # DICOM
            main_window.dicom = dcm.read_file(file_name, force=True)
            main_window.images = main_window.dicom.pixel_array
            if main_window.images.ndim == 4:  # 3 channel input
                main_window.images = main_window.images[:, :, :, 0]
            parse_dicom(main_window)
        except AttributeError:
            try:  # NIfTi
                main_window.images = sitk.GetArrayFromImage(sitk.ReadImage(file_name))
                main_window.file_name = main_window.file_name.split('_')[0]  # remove _img.nii suffix
            except:
                ErrorMessage(
                    main_window, 'File is not a valid IVUS file and could not be loaded (DICOM or NIfTi supported)'
                )
                return None

        main_window.file_name = os.path.splitext(file_name)[0]  # remove file extension
        main_window.metadata['num_frames'] = main_window.images.shape[0]
        main_window.display_slider.setMaximum(main_window.metadata['num_frames'] - 1)
        
        # Initialize 'eem' key
        main_window.data['eem'] = [[None] * main_window.metadata['num_frames'], [None] * main_window.metadata['num_frames']]

        success = read_contours(main_window, main_window.file_name)
        if success:
            main_window.segmentation = True
            try:
                main_window.gated_frames_dia = [
                    frame
                    for frame in range(main_window.metadata['num_frames'])
                    if main_window.data['phases'][frame] == 'D'
                ]
                main_window.gated_frames_sys = [
                    frame
                    for frame in range(main_window.metadata['num_frames'])
                    if main_window.data['phases'][frame] == 'S'
                ]
                main_window.gated_frames = main_window.gated_frames_dia
            except KeyError:  # old contour files may not have phases attribute
                pass
        else:  # initialise empty containers
            for key in [
                'plaque_frames',
                'lumen_area',
                'lumen_circumf',
                'longest_distance',
                'shortest_distance',
                'elliptic_ratio',
                'vector_length',
                'vector_angle',
                'eem',
            ]:
                main_window.data[key] = [0] * main_window.metadata['num_frames']
            main_window.data['phases'] = ['-'] * main_window.metadata['num_frames']
            for key in ['lumen_centroid', 'farthest_point', 'nearest_point', 'lumen']:
                main_window.data[key] = (
                    [[] for _ in range(main_window.metadata['num_frames'])],
                    [[] for _ in range(main_window.metadata['num_frames'])],
                )
            main_window.data['measures'] = [[None, None] for _ in range(main_window.metadata['num_frames'])]
            main_window.data['measure_lengths'] = [[np.nan, np.nan] for _ in range(main_window.metadata['num_frames'])]
            main_window.data['reference'] = [None] * main_window.metadata['num_frames']
            main_window.data['gating_signal'] = {}
            main_window.display.set_data(main_window.data['lumen'], main_window.images)
            main_window.display.set_data_eem(main_window.data['eem'], main_window.images)

        main_window.image_displayed = True
        main_window.display_slider.setValue(main_window.metadata['num_frames'] - 1)
    main_window.status_bar.showMessage(main_window.waiting_status)
