import os
import json
import glob

import numpy as np
from loguru import logger

from version import version_file_str
from gui.popup_windows.message_boxes import ErrorMessage
from input_output.read_xml import read_xml
from input_output.write_xml import write_xml


def read_contours(main_window, file_name=None):
    """Reads contours saved in json/xml format and displays the contours in the graphics scene"""
    success = False
    json_files = glob.glob(f'{file_name}_contours*.json')
    xml_files = glob.glob(f'{file_name}_contours*.xml')

    if not main_window.config.save.use_xml_files and json_files:  # json files have priority over xml unless desired
        newest_json = max(json_files)  # find file with most recent version
        logger.info(f'Current version is {version_file_str}, file found with most recent version is {newest_json}')
        with open(newest_json, 'r') as in_file:
            main_window.data = json.load(in_file)
        if 'measures' not in main_window.data:  # added in version 0.4.5
            main_window.data['measures'] = [[None, None] for _ in range(main_window.metadata['num_frames'])]
        if 'reference' not in main_window.data:  # added in version 0.7.3
            main_window.data['reference'] = [None] * main_window.metadata['num_frames']
        if 'gating_signal' not in main_window.data:  # added in version 0.7.4
            main_window.data['gating_signal'] = {}
        if 'eem' not in main_window.data: # added in version 0.8.0
            main_window.data['eem'] = [[None] * main_window.metadata['num_frames'], [None] * main_window.metadata['num_frames']]
        success = True

    elif xml_files:
        newest_xml = max(xml_files)  # find file with most recent version
        logger.info(f'Current version is {version_file_str}, file found with most recent version is {newest_xml}')
        read_xml(main_window, newest_xml)
        main_window.data['lumen'] = map_to_list(main_window.data['lumen'])
        for key in [
            'lumen_area',
            'lumen_circumf',
            'longest_distance',
            'shortest_distance',
            'elliptic_ratio',
            'vector_length',
            'vector_angle',
        ]:
            main_window.data[key] = [0] * main_window.metadata[
                'num_frames'
            ]  # initialise empty containers for data not stored in xml
        for key in ['lumen_centroid', 'farthest_point', 'nearest_point']:
            main_window.data[key] = (
                [[] for _ in range(main_window.metadata['num_frames'])],
                [[] for _ in range(main_window.metadata['num_frames'])],
            )  # initialise empty containers for data not stored in xml
        success = True

    if success:
        main_window.contours_drawn = True
        main_window.display.set_data(main_window.data['lumen'], main_window.images)
        main_window.display.set_data_eem(main_window.data['eem'], main_window.images)
        main_window.hide_contours_box.setChecked(False)

    return success


def write_contours(main_window):
    """Writes contours to a json/xml file"""

    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot write contours before reading input file')
        return

    if main_window.config.save.use_xml_files:
        # reformat data for compatibility with write_xml function
        x, y = [], []
        for frame in range(main_window.metadata['num_frames']):
            if frame < len(main_window.data['lumen'][0]):
                new_x_lumen = main_window.data['lumen'][0][frame]
                new_y_lumen = main_window.data['lumen'][1][frame]
            else:
                new_x_lumen = []
                new_y_lumen = []

            x.append(new_x_lumen)
            y.append(new_y_lumen)

        write_xml(
            x,
            y,
            main_window.images.shape,
            main_window.metadata['resolution'],
            main_window.ivusPullbackRate,
            main_window.data['phases'],
            main_window.file_name,
        )
    else:
        with open(os.path.join(main_window.file_name + f'_contours_{version_file_str}.json'), 'w') as out_file:
            json.dump(main_window.data, out_file)


def map_to_list(contours):
    """Converts map to list"""
    x, y = contours
    x = [list(x[i]) for i in range(len(x))]
    y = [list(y[i]) for i in range(len(y))]

    return (x, y)


def save_gated_images(main_window, file_name=None):
    """Saves diastolic and systolic images as a 3D numpy array"""
    if not main_window.image_displayed:
        ErrorMessage(main_window, 'Cannot save gated images before reading the input file.')
        return

    diastolic_images = []
    systolic_images = []

    for frame in range(main_window.metadata['num_frames']):
        if main_window.data['phases'][frame] == 'D':
            diastolic_images.append(main_window.images[frame])
        elif main_window.data['phases'][frame] == 'S':
            systolic_images.append(main_window.images[frame])

    diastolic_images = np.array(diastolic_images)
    systolic_images = np.array(systolic_images)

    out_path_diastolic = os.path.splitext(main_window.file_name)[0] + '_diastolic.npy'
    out_path_systolic = os.path.splitext(main_window.file_name)[0] + '_systolic.npy'
    np.save(out_path_diastolic, diastolic_images)
    np.save(out_path_systolic, systolic_images)
