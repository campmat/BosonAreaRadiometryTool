from PyQt5 import QtWidgets, QtCore, QtGui, uic

import sys
import os
import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import cv2

try:
    from flirpy.camera.boson import Boson
except Exception:  # Allows GUI development on PCs without FLIR/flirpy installed
    Boson = None


Point = Tuple[int, int]


def resource_path(relative_path: str) -> str:
    """Get absolute path to resource, works in development and PyInstaller bundle."""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


def find_ui_file() -> str:
    """Find the UI file both in the project tree and during local testing."""
    candidates = [
        resource_path("ui/window.ui"),
        resource_path("window.ui"),
        os.path.join(os.path.dirname(__file__), "ui", "window.ui"),
        os.path.join(os.path.dirname(__file__), "window.ui"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Could not find window.ui. Expected ./ui/window.ui or ./window.ui")


def frame_to_temperature_c(frame: np.ndarray) -> np.ndarray:
    """Convert Boson TLinear frame to Celsius. Assumption: raw value = Kelvin * 100."""
    return frame.astype(np.float32) / 100.0 - 273.15


def normalize_for_display(temp_img: np.ndarray) -> np.ndarray:
    """Normalize temperature image to 8-bit grayscale for display only."""
    display = cv2.normalize(temp_img, None, 0, 255, cv2.NORM_MINMAX)
    return display.astype(np.uint8)


def qcolor_to_bgr(color: QtGui.QColor) -> Tuple[int, int, int]:
    return color.blue(), color.green(), color.red()


def random_roi_color() -> QtGui.QColor:
    return QtGui.QColor(random.randint(40, 255), random.randint(40, 255), random.randint(40, 255))


@dataclass
class Roi:
    roi_type: str
    points: List[Point]
    mask: np.ndarray
    color: QtGui.QColor = field(default_factory=random_roi_color)

    @property
    def area_px(self) -> int:
        return int(np.count_nonzero(self.mask))

    def contains(self, x: int, y: int) -> bool:
        if 0 <= y < self.mask.shape[0] and 0 <= x < self.mask.shape[1]:
            return bool(self.mask[y, x])
        return False


class RoiStatisticsDialog(QtWidgets.QDialog):
    def __init__(self, rois: List[Roi], temp_img: Optional[np.ndarray], parent=None):
        super().__init__(parent)
        self.rois = rois
        self.temp_img = temp_img
        self.setWindowTitle("ROI temperature statistics")
        self.resize(760, 360)

        layout = QtWidgets.QVBoxLayout(self)
        self.table = QtWidgets.QTableWidget(self)
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Type", "Area [px²]", "Min [°C]", "Max [°C]", "Mean [°C]", "STD [°C]"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        layout.addWidget(self.table)

        self.populate(rois, temp_img)

    def update_statistics(self, temp_img: Optional[np.ndarray]) -> None:
        """Refresh statistics while the dialog is open and camera frames are updating."""
        self.temp_img = temp_img
        self.populate(self.rois, self.temp_img)

    def populate(self, rois: List[Roi], temp_img: Optional[np.ndarray]) -> None:
        self.table.setRowCount(len(rois))
        for row, roi in enumerate(rois):
            values = [roi.roi_type, str(roi.area_px)]

            if temp_img is not None and temp_img.shape[:2] == roi.mask.shape:
                roi_values = temp_img[roi.mask > 0]
                if roi_values.size > 0:
                    values.extend([
                        f"{np.min(roi_values):.2f}",
                        f"{np.max(roi_values):.2f}",
                        f"{np.mean(roi_values):.2f}",
                        f"{np.std(roi_values):.2f}",
                    ])
                else:
                    values.extend(["—", "—", "—", "—"])
            else:
                values.extend(["—", "—", "—", "—"])

            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table.setItem(row, col, item)


class BARTWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(find_ui_file(), self)

        self.active_tool: Optional[str] = None
        self.current_points: List[Point] = []
        self.current_hover_point: Optional[Point] = None
        self.rois: List[Roi] = []
        self.statistics_dialogs: List[RoiStatisticsDialog] = []

        self.camera = None
        self.current_temp_img: Optional[np.ndarray] = None
        self.current_display_img: Optional[np.ndarray] = None
        self.image_shape = (320, 320)  # Updated after first frame

        self.scene = QtWidgets.QGraphicsScene(self)
        self.pixmap_item = QtWidgets.QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setMouseTracking(True)
        self.graphicsView.viewport().setMouseTracking(True)
        self.graphicsView.viewport().installEventFilter(self)
        self.graphicsView.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

        self.configure_buttons()
        self.configure_roi_table()
        self.start_camera()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # ---------------------------- setup ----------------------------
    def configure_buttons(self) -> None:
        self.tool_buttons = {
            "Rectangle": self.rectangleButton,
            "Ellipse": self.ellipseButton,
            "Polygon": self.polygonButton,
        }
        for tool_name, button in self.tool_buttons.items():
            button.setCheckable(True)
            button.clicked.connect(lambda checked, name=tool_name: self.activate_tool(name))

    def configure_roi_table(self) -> None:
        self.tableWidget.setColumnCount(3)
        self.tableWidget.setHorizontalHeaderLabels(["Color", "Type", "Area [px²]"])
        self.tableWidget.horizontalHeader().setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        self.tableWidget.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        self.tableWidget.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.tableWidget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.tableWidget.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tableWidget.cellDoubleClicked.connect(self.open_statistics_dialog)

    def start_camera(self) -> None:
        if Boson is None:
            self.statusBar().showMessage("FLIR camera module not available. Using simulated frame.")
            return
        try:
            self.camera = Boson()
            self.statusBar().showMessage("FLIR Boson connected.")
        except Exception as exc:
            self.camera = None
            self.statusBar().showMessage(f"Could not open FLIR Boson. Using simulated frame. Reason: {exc}")

    # ---------------------------- tool state ----------------------------
    def activate_tool(self, tool_name: str) -> None:
        """Activate one drawing tool and reset any unfinished shape."""
        self.active_tool = tool_name
        self.current_points.clear()
        self.current_hover_point = None

        for name, button in self.tool_buttons.items():
            is_active = name == tool_name
            button.blockSignals(True)
            button.setChecked(is_active)
            button.setEnabled(not is_active)  # visually shows active and prevents duplicate click
            button.blockSignals(False)

        self.statusBar().showMessage(f"Active ROI tool: {tool_name}")
        self.render_scene()

    def deactivate_tool(self) -> None:
        self.active_tool = None
        self.current_points.clear()
        self.current_hover_point = None
        for button in self.tool_buttons.values():
            button.blockSignals(True)
            button.setChecked(False)
            button.setEnabled(True)
            button.blockSignals(False)
        self.render_scene()

    # ---------------------------- camera and rendering ----------------------------
    def update_frame(self) -> None:
        try:
            if self.camera is not None:
                raw_frame = self.camera.grab()
                self.current_temp_img = frame_to_temperature_c(raw_frame)
            else:
                self.current_temp_img = self.simulated_temperature_frame()
        except Exception as exc:
            self.current_temp_img = self.simulated_temperature_frame()
            self.statusBar().showMessage(f"Camera read failed. Using simulated frame. Reason: {exc}")

        self.image_shape = self.current_temp_img.shape[:2]
        display_gray = normalize_for_display(self.current_temp_img)
        self.current_display_img = cv2.cvtColor(display_gray, cv2.COLOR_GRAY2BGR)
        self.render_scene()
        self.update_open_statistics_dialogs()

    def simulated_temperature_frame(self) -> np.ndarray:
        h, w = self.image_shape
        y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
        return 22.0 + 8.0 * x + 4.0 * y + np.random.normal(0, 0.08, (h, w)).astype(np.float32)

    def render_scene(self) -> None:
        if self.current_display_img is None:
            return

        image = self.current_display_img.copy()
        self.draw_committed_rois(image)
        self.draw_preview(image)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, channels = rgb.shape
        bytes_per_line = channels * w
        qimg = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()
        self.pixmap_item.setPixmap(QtGui.QPixmap.fromImage(qimg))
        self.scene.setSceneRect(0, 0, w, h)
        self.graphicsView.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def draw_committed_rois(self, image: np.ndarray) -> None:
        overlay = image.copy()
        for roi in self.rois:
            bgr = qcolor_to_bgr(roi.color)
            overlay[roi.mask > 0] = bgr
            self.draw_roi_outline(image, roi.roi_type, roi.points, bgr, thickness=2)
            for point in roi.points:
                cv2.circle(image, point, 3, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.30, image, 0.70, 0, dst=image)

        # Redraw outlines after alpha blending so edges stay fully visible.
        for roi in self.rois:
            self.draw_roi_outline(image, roi.roi_type, roi.points, qcolor_to_bgr(roi.color), thickness=2)

    def draw_preview(self, image: np.ndarray) -> None:
        preview_points = list(self.current_points)
        if self.current_hover_point is not None:
            preview_points.append(self.adjust_hover_point(self.current_hover_point))

        for point in preview_points:
            cv2.circle(image, point, 4, (0, 0, 255), -1, cv2.LINE_AA)

        if self.active_tool and len(preview_points) >= 2:
            self.draw_roi_outline(image, self.active_tool, preview_points, (0, 0, 255), thickness=1)

    def draw_roi_outline(self, image: np.ndarray, roi_type: str, points: List[Point], color, thickness: int = 1) -> None:
        if roi_type == "Rectangle" and len(points) >= 2:
            p1, p2 = points[0], points[1]
            cv2.rectangle(image, p1, p2, color, thickness, cv2.LINE_AA)
        elif roi_type == "Ellipse" and len(points) >= 2:
            center, axis_point = points[0], points[1]
            a = max(1, int(distance(center, axis_point)))
            angle = math.degrees(math.atan2(axis_point[1] - center[1], axis_point[0] - center[0]))
            b = a
            if len(points) >= 3:
                b = max(1, int(perpendicular_distance(points[2], center, axis_point)))
            cv2.ellipse(image, center, (a, b), angle, 0, 360, color, thickness, cv2.LINE_AA)
        elif roi_type == "Polygon" and len(points) >= 2:
            # Close the outline only after the polygon has at least 3 points.
            # During two-point preview it remains an open segment.
            is_closed = len(points) >= 3
            cv2.polylines(image, [np.array(points, dtype=np.int32)], is_closed, color, thickness, cv2.LINE_AA)

    # ---------------------------- interaction ----------------------------
    def eventFilter(self, obj, event):
        if obj is self.graphicsView.viewport():
            if event.type() == QtCore.QEvent.MouseMove:
                point = self.view_to_image_point(event.pos())
                if point is not None:
                    self.current_hover_point = point
                    self.render_scene()
                return False

            if event.type() == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton:
                point = self.view_to_image_point(event.pos())
                if point is not None and self.active_tool:
                    self.add_current_point(point)
                    return True

            if event.type() == QtCore.QEvent.MouseButtonDblClick:
                point = self.view_to_image_point(event.pos())
                if point is not None:
                    for roi in reversed(self.rois):
                        if roi.contains(*point):
                            self.open_statistics_dialog()
                            return True

            if event.type() == QtCore.QEvent.Leave:
                self.current_hover_point = None
                self.render_scene()
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
            if self.active_tool == "Polygon" and len(self.current_points) >= 3:
                self.commit_current_roi()
                return
            if self.active_tool == "Ellipse" and len(self.current_points) >= 2:
                self.commit_current_roi()
                return
        elif event.key() == QtCore.Qt.Key_Escape:
            self.deactivate_tool()
            return
        elif event.key() == QtCore.Qt.Key_Delete:
            self.delete_selected_rois()
            return
        super().keyPressEvent(event)

    def view_to_image_point(self, view_pos: QtCore.QPoint) -> Optional[Point]:
        scene_pos = self.graphicsView.mapToScene(view_pos)
        x, y = int(round(scene_pos.x())), int(round(scene_pos.y()))
        h, w = self.image_shape
        if 0 <= x < w and 0 <= y < h:
            return x, y
        return None

    def add_current_point(self, point: Point) -> None:
        point = self.adjust_hover_point(point)
        self.current_points.append(point)

        if self.active_tool == "Rectangle" and len(self.current_points) == 2:
            self.commit_current_roi()
        elif self.active_tool == "Ellipse" and len(self.current_points) == 3:
            self.commit_current_roi()
        elif self.active_tool == "Polygon":
            self.statusBar().showMessage("Polygon: click more points or press ENTER to finish.")

        self.render_scene()

    def adjust_hover_point(self, point: Point) -> Point:
        """Force square preview/selection for rectangles when Shift is held."""
        if self.active_tool != "Rectangle" or len(self.current_points) != 1:
            return point
        if not (QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.ShiftModifier):
            return point

        x0, y0 = self.current_points[0]
        dx, dy = point[0] - x0, point[1] - y0
        side = max(abs(dx), abs(dy))
        x = x0 + side * (1 if dx >= 0 else -1)
        y = y0 + side * (1 if dy >= 0 else -1)
        h, w = self.image_shape
        return int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))

    # ---------------------------- ROI creation ----------------------------
    def commit_current_roi(self) -> None:
        if not self.active_tool:
            return
        mask = self.create_mask(self.active_tool, self.current_points)
        if mask is None or np.count_nonzero(mask) == 0:
            self.statusBar().showMessage("ROI was not created because its area is zero.")
            self.current_points.clear()
            return

        roi = Roi(self.active_tool, list(self.current_points), mask, random_roi_color())
        self.rois.append(roi)
        self.add_roi_table_row(roi)
        self.statusBar().showMessage(f"Added {roi.roi_type} ROI, area = {roi.area_px} px²")
        self.current_points.clear()
        self.current_hover_point = None

    def create_mask(self, roi_type: str, points: List[Point]) -> Optional[np.ndarray]:
        h, w = self.image_shape
        mask = np.zeros((h, w), dtype=np.uint8)

        if roi_type == "Rectangle" and len(points) >= 2:
            (x1, y1), (x2, y2) = points[0], points[1]
            x_min, x_max = sorted((x1, x2))
            y_min, y_max = sorted((y1, y2))
            mask[y_min:y_max + 1, x_min:x_max + 1] = 255

        elif roi_type == "Ellipse" and len(points) >= 2:
            center, axis_point = points[0], points[1]
            a = max(1, int(distance(center, axis_point)))
            b = a
            if len(points) >= 3:
                b = max(1, int(perpendicular_distance(points[2], center, axis_point)))
            angle = math.degrees(math.atan2(axis_point[1] - center[1], axis_point[0] - center[0]))
            cv2.ellipse(mask, center, (a, b), angle, 0, 360, 255, -1, cv2.LINE_AA)

        elif roi_type == "Polygon" and len(points) >= 3:
            cv2.fillPoly(mask, [np.array(points, dtype=np.int32)], 255, cv2.LINE_AA)

        else:
            return None

        return mask

    def add_roi_table_row(self, roi: Roi) -> None:
        row = self.tableWidget.rowCount()
        self.tableWidget.insertRow(row)

        color_button = QtWidgets.QPushButton()
        color_button.setFixedSize(34, 22)
        color_button.clicked.connect(lambda checked=False, r=roi, b=color_button: self.change_roi_color(r, b))
        self.update_color_button_style(color_button, roi.color)
        self.tableWidget.setCellWidget(row, 0, color_button)

        type_item = QtWidgets.QTableWidgetItem(roi.roi_type)
        type_item.setTextAlignment(QtCore.Qt.AlignCenter)
        area_item = QtWidgets.QTableWidgetItem(str(roi.area_px))
        area_item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.tableWidget.setItem(row, 1, type_item)
        self.tableWidget.setItem(row, 2, area_item)

    def delete_selected_rois(self) -> None:
        """Delete selected ROI rows and their corresponding shapes."""
        selected_rows = sorted(
            {index.row() for index in self.tableWidget.selectedIndexes()},
            reverse=True,
        )
        if not selected_rows:
            return

        for row in selected_rows:
            if 0 <= row < len(self.rois):
                del self.rois[row]
                self.tableWidget.removeRow(row)

        self.render_scene()
        self.update_open_statistics_dialogs()
        self.statusBar().showMessage(f"Deleted {len(selected_rows)} ROI(s).")

    def update_open_statistics_dialogs(self) -> None:
        """Refresh all visible statistics dialogs and forget dialogs that were closed."""
        still_open = []
        for dialog in self.statistics_dialogs:
            if dialog.isVisible():
                dialog.update_statistics(self.current_temp_img)
                still_open.append(dialog)
        self.statistics_dialogs = still_open

    def update_color_button_style(self, button: QtWidgets.QPushButton, color: QtGui.QColor) -> None:
        button.setStyleSheet(
            f"QPushButton {{"
            f"background-color: rgba({color.red()}, {color.green()}, {color.blue()}, 90);"
            f"border: 2px solid rgb({color.red()}, {color.green()}, {color.blue()});"
            f"border-radius: 8px;"
            f"}}"
        )

    def change_roi_color(self, roi: Roi, button: QtWidgets.QPushButton) -> None:
        color = QtWidgets.QColorDialog.getColor(roi.color, self, "Select ROI color")
        if color.isValid():
            roi.color = color
            self.update_color_button_style(button, color)
            self.render_scene()

    def open_statistics_dialog(self, *args) -> None:
        # Non-modal dialog so it can update live while the camera stream continues.
        dialog = RoiStatisticsDialog(self.rois, self.current_temp_img, self)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        dialog.destroyed.connect(lambda *_: self.update_open_statistics_dialogs())
        self.statistics_dialogs.append(dialog)
        dialog.show()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.camera is not None:
            try:
                self.camera.close()
            except Exception:
                pass
        event.accept()


def distance(p1: Point, p2: Point) -> float:
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def perpendicular_distance(point: Point, line_start: Point, line_end: Point) -> float:
    """Distance from point to the infinite line through line_start and line_end."""
    x0, y0 = point
    x1, y1 = line_start
    x2, y2 = line_end
    denominator = math.hypot(x2 - x1, y2 - y1)
    if denominator == 0:
        return 1.0
    return abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / denominator


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = BARTWindow()
    window.show()
    sys.exit(app.exec_())
