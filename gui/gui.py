from loguru import logger
from PyQt5.QtWidgets import (
    QMainWindow,
    QMenuBar,
    QSplitter,
    QTableWidget,
    QStatusBar,
)
from PyQt5.QtCore import QTimer

from gui.left_half.left_half import LeftHalf
from gui.right_half.right_half import RightHalf
from gui.shortcuts import init_shortcuts, init_menu
from input_output.contours_io import write_contours
from gating.contour_based_gating import ContourBasedGating
from segmentation.predict import Predict


class Master(QMainWindow):
    """Main Window Class"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.file_name = None  # Ensure file_name is initialized
        self.autosave_interval = config.save.autosave_interval
        self.contour_based_gating = ContourBasedGating(self)
        self.predictor = Predict(self)
        self.image_displayed = False
        self.contours_drawn = False
        self.hide_contours = False
        self.hide_special_points = False
        self.colormap_enabled = False
        self.filter = None
        self.tmp_lumen_x = []  # for Ctrl+Z
        self.tmp_lumen_y = []
        self.tmp_eem_x = []
        self.tmp_eem_y = []
        self.gated_frames = []
        self.gated_frames_dia = []
        self.gated_frames_sys = []
        self.data = {
            'lumen': [[], []],
            'eem': [[], []],  # Initialize 'eem' key
        }
        self.metadata = {}  # metadata used outside of read_image (not saved to JSON file)
        self.images = None
        self.diastole_color = (39, 69, 219)
        self.diastole_color_plt = tuple(x / 255 for x in self.diastole_color)  # for matplotlib
        self.systole_color = (209, 55, 38)
        self.systole_color_plt = tuple(x / 255 for x in self.systole_color)
        self.measure_colors = ['red', 'cyan']
        self.reference_color = 'yellow'
        self.waiting_status = 'Waiting for user input...'
        self.init_gui()
        init_shortcuts(self)

    def init_gui(self):
        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)
        self.file_name = "default_file_name"  # Initialize file_name with a default value
        init_menu(self)
        self.metadata_table = QTableWidget()

        self.status_bar = QStatusBar(self)
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(self.waiting_status)

        main_window_splitter = QSplitter()
        main_window_splitter.addWidget(LeftHalf(self)())
        main_window_splitter.addWidget(RightHalf(self)())

        self.setWindowTitle('AAOCA Segmentation Tool')
        self.setCentralWidget(main_window_splitter)
        self.showMaximized()

        timer = QTimer(self)
        timer.timeout.connect(self.auto_save)
        timer.start(self.autosave_interval)  # autosave interval in milliseconds

    def auto_save(self):
        if self.image_displayed:
            write_contours(self)
