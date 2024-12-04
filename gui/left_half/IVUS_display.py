import math
import cv2

import numpy as np
from loguru import logger
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsTextItem
from PyQt5.QtCore import Qt, QLineF, QPointF
from PyQt5.QtGui import QPixmap, QImage, QColor, QFont, QPen
from shapely.geometry import Polygon

from gui.utils.geometry import Point, Spline, get_qt_pen
from gui.right_half.longitudinal_view import Marker
from report.report import compute_polygon_metrics, farthest_points, closest_points
from segmentation.segment import downsample


class IVUSDisplay(QGraphicsView):
    """
    Displays images and contours and allows the user to add and manipulate contours.
    """

    def __init__(self, main_window):
        super(IVUSDisplay, self).__init__()
        self.main_window = main_window
        config = main_window.config
        self.n_interactive_points = config.display.n_interactive_points
        self.n_points_contour = config.display.n_points_contour
        self.image_size = config.display.image_size
        self.windowing_sensitivity = config.display.windowing_sensitivity
        self.contour_thickness = config.display.contour_thickness
        self.point_thickness = config.display.point_thickness
        self.point_radius = config.display.point_radius
        self.color_contour = config.display.color_contour
        self.alpha_contour = config.display.alpha_contour
        self.graphics_scene = QGraphicsScene(self)
        self.points_to_draw = []
        self.contour_points = []
        self.frame = 0
        self.contour_mode = False
        self.contour_drawn = False
        self.current_contour = None  # entire contour (not only knotpoints), needed for elliptic ratio
        self.new_spline = None
        self.active_point = None
        self.active_point_index = None
        self.measure_index = None
        self.measure_colors = self.main_window.measure_colors
        self.reference_mode = False

        # Store initial window level and window width (full width, middle level)
        self.initial_window_level = 128  # window level is the center which determines the brightness of the image
        self.initial_window_width = 256  # window width is the range of pixel values that are displayed
        self.window_level = self.initial_window_level
        self.window_width = self.initial_window_width

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        image = QGraphicsPixmapItem(QPixmap(self.image_size, self.image_size))
        self.graphics_scene.addItem(image)
        self.setScene(self.graphics_scene)

    def set_data(self, lumen, images):
        num_frames = images.shape[0]
        self.image_width = images.shape[1]
        self.scaling_factor = self.image_size / images.shape[1]
        self.main_window.data['lumen'] = lumen
        self.full_contours = [
            (
                Spline(
                    [lumen[0][frame], lumen[1][frame]],
                    self.n_points_contour,
                    self.contour_thickness,
                    self.color_contour,
                    self.alpha_contour,
                ).get_unscaled_contour(
                    scaling_factor=1
                )  # data is not yet scaled at read, hence scaling_factor=1
                if lumen[0][frame]
                else None
            )
            for frame in range(num_frames)
        ]
        self.images = images
        self.main_window.longitudinal_view.set_data(self.images, self.full_contours)
        self.display_image(update_image=True, update_contours=True, update_phase=True)

    def display_image(self, update_image=False, update_contours=False, update_phase=False):
        """Clears scene and displays current image and contours"""
        image_types = (QGraphicsPixmapItem, Marker)
        if update_image:
            [
                self.graphics_scene.removeItem(item)
                for item in self.graphics_scene.items()
                if isinstance(item, image_types)
            ]  # clear previous scene
            self.active_point = None
            self.active_point_index = None

            # Calculate lower and upper bounds for the adjusted window level and window width
            lower_bound = self.window_level - self.window_width / 2
            upper_bound = self.window_level + self.window_width / 2

            # Clip and normalize pixel values
            normalised_data = np.clip(self.images[self.frame, :, :], lower_bound, upper_bound)
            normalised_data = ((normalised_data - lower_bound) / (upper_bound - lower_bound) * 255).astype(np.uint8)
            height, width = normalised_data.shape

            if self.main_window.filter == 0:
                normalised_data = cv2.medianBlur(normalised_data, 5)
            elif self.main_window.filter == 1:
                normalised_data = cv2.GaussianBlur(normalised_data, (5, 5), 0)
            elif self.main_window.filter == 2:
                normalised_data = cv2.bilateralFilter(normalised_data, 9, 75, 75)

            if self.main_window.colormap_enabled:
                # Apply an orange-blue colormap
                colormap = cv2.applyColorMap(normalised_data, cv2.COLORMAP_COOL)
                q_image = QImage(colormap.data, width, height, width * 3, QImage.Format.Format_RGB888).scaled(
                    self.image_size, self.image_size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
                )
            else:
                q_image = QImage(normalised_data.data, width, height, width, QImage.Format.Format_Grayscale8).scaled(
                    self.image_size, self.image_size, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
                )

            image = QGraphicsPixmapItem(QPixmap.fromImage(q_image))
            self.graphics_scene.addItem(image)

            self.main_window.longitudinal_view.update_marker(self.frame)
            marker = Marker(
                (self.image_width // 2) * self.scaling_factor,
                0,
                (self.image_width // 2) * self.scaling_factor,
                height * self.scaling_factor,
            )
            self.graphics_scene.addItem(marker)

        old_contours = [item for item in self.graphics_scene.items() if not isinstance(item, image_types)]
        [
            self.graphics_scene.removeItem(item)
            for item in self.graphics_scene.items()
            if not isinstance(item, image_types)
        ]  # clear previous scene
        if self.main_window.hide_contours:
            self.main_window.longitudinal_view.hide_lview_contours()
        else:
            if update_contours:
                self.draw_contour(self.main_window.data['lumen'])
                self.draw_measure()
                self.draw_reference()
                if self.main_window.data['lumen'][0][self.frame] and self.current_contour.full_contour[0] is not None:
                    lumen_x, lumen_y = self.current_contour.get_unscaled_contour(self.scaling_factor)
                    polygon = Polygon([(x, y) for x, y in zip(lumen_x, lumen_y)])
                    lumen_area, lumen_circumf, _, _ = compute_polygon_metrics(self.main_window, polygon, self.frame)
                    longest_distance, farthest_point_x, farthest_point_y = farthest_points(
                        self.main_window, polygon.exterior.coords, self.frame
                    )
                    shortest_distance, closest_point_x, closest_point_y = closest_points(
                        self.main_window, polygon, self.frame
                    )
                    if not self.main_window.hide_special_points:
                        self.graphics_scene.addLine(
                            QLineF(
                                farthest_point_x[0] * self.scaling_factor,
                                farthest_point_y[0] * self.scaling_factor,
                                farthest_point_x[1] * self.scaling_factor,
                                farthest_point_y[1] * self.scaling_factor,
                            ),
                            QPen(Qt.yellow, self.point_thickness * 2),
                        )
                        self.graphics_scene.addLine(
                            QLineF(
                                closest_point_x[0] * self.scaling_factor,
                                closest_point_y[0] * self.scaling_factor,
                                closest_point_x[1] * self.scaling_factor,
                                closest_point_y[1] * self.scaling_factor,
                            ),
                            QPen(Qt.yellow, self.point_thickness * 2),
                        )

                    elliptic_ratio = (longest_distance / shortest_distance) if shortest_distance != 0 else 0
                    frame_metrics_text = QGraphicsTextItem(
                        f'Lumen area:\t\t{round(lumen_area, 2)} (mm\N{SUPERSCRIPT TWO})\n'
                        f'Lumen circ:\t\t{round(lumen_circumf, 2)} (mm)\n'
                        f'Elliptic ratio:\t\t{round(elliptic_ratio, 2)}\n'
                        f'Longest distance:\t{round(longest_distance, 2)} (mm)\n'
                        f'Shortest distance:\t{round(shortest_distance, 2)} (mm)'
                    )
                    frame_metrics_text.setFont(QFont('Helvetica', int(self.image_size / 50)))
                    self.graphics_scene.addItem(frame_metrics_text)
                    if not update_phase:
                        self.graphics_scene.addItem(self.phase_text)
            else:  # re-draw old elements to put them in foreground
                [self.graphics_scene.addItem(item) for item in old_contours]

        if update_phase:
            if self.main_window.data['phases'][self.frame] == 'D':
                phase = 'Diastole'
                color = QColor(
                    self.main_window.diastole_color[0],
                    self.main_window.diastole_color[1],
                    self.main_window.diastole_color[2],
                )
            elif self.main_window.data['phases'][self.frame] == 'S':
                phase = 'Systole'
                color = QColor(
                    self.main_window.systole_color[0],
                    self.main_window.systole_color[1],
                    self.main_window.systole_color[2],
                )
            else:
                phase = ''
                color = Qt.white
            self.phase_text = QGraphicsTextItem(phase)
            self.phase_text.setDefaultTextColor(color)
            self.phase_text.setX(self.image_size - self.image_size / 3.75)
            self.phase_text.setFont(QFont('Helvetica', int(self.image_size / 50), QFont.Bold))
            self.graphics_scene.addItem(self.phase_text)

    def draw_contour(self, lumen):
        """Adds lumen contours to scene"""

        if lumen[0][self.frame]:
            lumen_x = [point * self.scaling_factor for point in lumen[0][self.frame]]
            lumen_y = [point * self.scaling_factor for point in lumen[1][self.frame]]
            self.current_contour = Spline(
                [lumen_x, lumen_y],
                self.n_points_contour,
                self.contour_thickness,
                self.color_contour,
                self.alpha_contour,
            )
            if self.current_contour.full_contour[0] is not None:
                self.contour_points = [
                    Point(
                        (self.current_contour.knot_points[0][i], self.current_contour.knot_points[1][i]),
                        self.point_thickness,
                        self.point_radius,
                        self.color_contour,
                        self.alpha_contour,
                    )
                    for i in range(len(self.current_contour.knot_points[0]) - 1)
                ]
                [self.graphics_scene.addItem(point) for point in self.contour_points]
                self.graphics_scene.addItem(self.current_contour)
                self.full_contours[self.frame] = self.current_contour.get_unscaled_contour(self.scaling_factor)
            else:
                logger.warning(f'Spline for frame {self.frame + 1} could not be interpolated')

    def add_contour(self, point):
        """Creates an interactive contour manually point by point"""

        if self.points_to_draw:
            start_point = self.points_to_draw[0].get_coords()
        else:
            self.contour_drawn = False
            start_point = (point.x(), point.y())

        if start_point[0] is None:  # occurs when Point has been deleted during draw (e.g. by RMB click)
            self.points_to_draw = []
            self.contour_mode = False
            self.main_window.setCursor(Qt.ArrowCursor)
            self.display_image(update_contours=True)
        else:
            if len(self.points_to_draw) > 3:  # start drawing spline after 3 points
                if not self.contour_drawn:
                    self.new_spline = Spline(
                        [
                            [point.get_coords()[0] for point in self.points_to_draw],
                            [point.get_coords()[1] for point in self.points_to_draw],
                        ],
                        self.n_points_contour,
                        self.contour_thickness,
                    )
                    self.graphics_scene.addItem(self.new_spline)
                    self.contour_drawn = True
                else:
                    self.new_spline.update(point, len(self.points_to_draw))

            if len(self.points_to_draw) > 1:  # check distance to start point, if close enough, close contour
                dist = math.sqrt((point.x() - start_point[0]) ** 2 + (point.y() - start_point[1]) ** 2)

                if dist < 20:
                    self.points_to_draw = []
                    if self.new_spline is not None:
                        downsampled = downsample(
                            ([self.new_spline.full_contour[0].tolist()], [self.new_spline.full_contour[1].tolist()]),
                            self.n_interactive_points,
                        )
                        self.main_window.data['lumen'][0][self.frame] = [
                            point / self.scaling_factor for point in downsampled[0]
                        ]
                        self.main_window.data['lumen'][1][self.frame] = [
                            point / self.scaling_factor for point in downsampled[1]
                        ]

                    self.stop_contour()
                    return

            self.points_to_draw.append(Point((point.x(), point.y()), self.point_thickness, self.point_radius))
            self.graphics_scene.addItem(self.points_to_draw[-1])

    def start_contour(self):
        self.measure_index = None
        self.main_window.setCursor(Qt.CrossCursor)
        self.contour_mode = True
        self.points_to_draw = []
        self.main_window.data['lumen'][0][self.frame] = []
        self.main_window.data['lumen'][1][self.frame] = []
        self.display_image(update_contours=True)  # clear previous contour

    def stop_contour(self):
        if self.main_window.image_displayed:
            self.contour_mode = False
            self.main_window.setCursor(Qt.ArrowCursor)
            self.display_image(update_contours=True)
            self.main_window.longitudinal_view.lview_contour(self.frame, self.full_contours[self.frame], update=True)

    def draw_measure(self):
        for index in range(2):
            if (
                self.main_window.data['measures'][self.frame][index] is not None
                and len(self.main_window.data['measures'][self.frame][index]) == 4
            ):
                first_point = QPointF(
                    self.main_window.data['measures'][self.frame][index][0],
                    self.main_window.data['measures'][self.frame][index][1],
                )
                second_point = QPointF(
                    self.main_window.data['measures'][self.frame][index][2],
                    self.main_window.data['measures'][self.frame][index][3],
                )
                self.main_window.data['measures'][self.frame][index] = None
                self.add_measure(first_point, index=index, new=False)
                self.add_measure(second_point, index=index, new=False)

    def add_measure(self, point, index=None, new=True):
        index = index if index is not None else self.measure_index
        new_point = Point((point.x(), point.y()), self.point_thickness, self.point_radius, self.measure_colors[index])
        self.graphics_scene.addItem(new_point)

        if self.main_window.data['measures'][self.frame][index] is None:
            self.main_window.data['measures'][self.frame][index] = [point.x(), point.y()]
        else:  # second point
            self.main_window.data['measures'][self.frame][index] += [point.x(), point.y()]
            line = QLineF(
                self.main_window.data['measures'][self.frame][index][0],
                self.main_window.data['measures'][self.frame][index][1],
                self.main_window.data['measures'][self.frame][index][2],
                self.main_window.data['measures'][self.frame][index][3],
            )
            length = round(line.length() * self.main_window.metadata["resolution"] / self.scaling_factor, 2)
            self.main_window.data['measure_lengths'][self.frame][index] = length
            length_text = QGraphicsTextItem(f'{length} mm')
            length_text.setPos(point.x(), point.y())
            self.graphics_scene.addItem(length_text)
            self.graphics_scene.addLine(line, get_qt_pen(self.measure_colors[index], self.point_thickness))
            if new:
                self.measure_index = None
                self.main_window.setCursor(Qt.ArrowCursor)

    def start_measure(self, index: int):
        if self.contour_mode:
            self.stop_contour()
        self.main_window.data['measures'][self.frame][index] = None  # reset this measure
        self.main_window.setCursor(Qt.CrossCursor)
        self.measure_index = index
        self.display_image(update_contours=True)

    def draw_reference(self):
        if self.main_window.data['reference'][self.frame] is not None:
            reference_point = self.main_window.data['reference'][self.frame]
            reference = Point(
                (reference_point[0], reference_point[1]),
                self.point_thickness,
                self.point_radius,
                self.main_window.reference_color,
            )
            self.graphics_scene.addItem(reference)
            text = QGraphicsTextItem('Reference')
            text.setPos(reference_point[0], reference_point[1])
            self.graphics_scene.addItem(text)

    def start_reference(self):
        self.reference_mode = True
        self.main_window.setCursor(Qt.CrossCursor)
        self.main_window.data['reference'][self.frame] = None
        self.display_image(update_contours=True)

    def update_display(self):
        self.display_image(update_image=True, update_contours=True, update_phase=True)

    def set_frame(self, value):
        self.frame = value
        self.current_contour = None
        self.stop_contour()
        if self.measure_index is not None:
            self.stop_measure(self.measure_index)

    def mousePressEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            pos = self.mapToScene(event.pos())
            if self.contour_mode:
                self.add_contour(pos)
            elif self.measure_index is not None:  # drawing measure
                self.add_measure(pos)
            elif self.reference_mode:
                self.main_window.data['reference'][self.frame] = [pos.x(), pos.y()]
                self.reference_mode = False
                self.main_window.setCursor(Qt.ArrowCursor)
                self.display_image(update_contours=True)
            else:
                # identify which point has been clicked
                items = self.items(event.pos())
                point = [item for item in items if isinstance(item, Point)]
                spline = [item for item in items if isinstance(item, Spline)]
                if point and point[0] in self.contour_points:
                    self.main_window.setCursor(Qt.BlankCursor)  # remove cursor for precise contour changes
                    # Convert mouse position to item position
                    # https://stackoverflow.com/questions/53627056/how-to-get-cursor-click-position-in-qgraphicsitem-coordinate-system
                    self.active_point_index = self.contour_points.index(point[0])
                    point[0].update_color()
                    self.active_point = point[0]
                elif spline:  # clicked on contour
                    path_index = self.current_contour.on_path(pos)
                    self.main_window.setCursor(Qt.BlankCursor)
                    self.active_point = Point(
                        (pos.x(), pos.y()),
                        self.point_thickness,
                        self.point_radius,
                        self.color_contour,
                        self.alpha_contour,
                    )
                    self.graphics_scene.addItem(self.active_point)
                    self.active_point.update_color()
                    self.active_point_index = self.current_contour.update(pos, self.active_point_index, path_index)

        elif event.buttons() == Qt.MouseButton.RightButton:
            self.mouse_x = event.x()
            self.mouse_y = event.y()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            if self.active_point_index is not None:
                item = self.active_point
                mouse_position = item.mapFromScene(self.mapToScene(event.pos()))
                new_point = item.update_pos(mouse_position)
                self.current_contour.update(new_point, self.active_point_index)

        elif event.buttons() == Qt.MouseButton.RightButton:
            self.setMouseTracking(True)
            # Right-click drag for adjusting window level and window width
            self.window_level += (event.x() - self.mouse_x) * self.windowing_sensitivity
            self.window_width += (event.y() - self.mouse_y) * self.windowing_sensitivity
            self.display_image(update_image=True)
            self.setMouseTracking(False)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:  # for some reason event.buttons() does not work here
            if self.active_point_index is not None:
                self.main_window.setCursor(Qt.ArrowCursor)
                item = self.active_point
                item.reset_color()

                self.main_window.data['lumen'][0][self.frame] = [
                    point / self.scaling_factor for point in self.current_contour.knot_points[0]
                ]
                self.main_window.data['lumen'][1][self.frame] = [
                    point / self.scaling_factor for point in self.current_contour.knot_points[1]
                ]
                self.display_image(update_contours=True)
                self.main_window.longitudinal_view.lview_contour(
                    self.frame, self.full_contours[self.frame], update=True
                )
                self.active_point_index = None
