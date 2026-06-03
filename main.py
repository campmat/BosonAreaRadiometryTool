from PyQt5 import QtWidgets, QtCore, QtGui, uic

import sys
import os
import math
import random
import json
import csv
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime

import numpy as np
import cv2

try:
    from flirpy.camera.boson import Boson
except Exception:  # Allows GUI development on PCs without FLIR/flirpy installed
    Boson = None


Point = Tuple[int, int]


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
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
        "/mnt/data/window(5).ui",  # useful only while testing in this ChatGPT sandbox
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Could not find window.ui. Expected ./ui/window.ui or ./window.ui")


def frame_to_temperature_c(frame: np.ndarray) -> np.ndarray:
    """Convert Boson TLinear frame to Celsius. Assumption: raw value = Kelvin * 100."""
    return frame.astype(np.float32) / 100.0 - 273.15


def temperature_c_to_raw_tlinear(temp_img: np.ndarray) -> np.ndarray:
    """Convert Celsius image back to Boson-like TLinear values for optional export."""
    return np.round((temp_img + 273.15) * 100.0).astype(np.uint16)


def normalize_for_display(temp_img: np.ndarray) -> np.ndarray:
    """Normalize temperature image to 8-bit grayscale for display only."""
    display = cv2.normalize(temp_img, None, 0, 255, cv2.NORM_MINMAX)
    return display.astype(np.uint8)


def qcolor_to_bgr(color: QtGui.QColor) -> Tuple[int, int, int]:
    return color.blue(), color.green(), color.red()


def qcolor_to_rgba_dict(color: QtGui.QColor) -> Dict[str, int]:
    return {"r": color.red(), "g": color.green(), "b": color.blue(), "a": color.alpha()}


def rgba_dict_to_qcolor(data: Dict[str, Any]) -> QtGui.QColor:
    return QtGui.QColor(int(data.get("r", 255)), int(data.get("g", 0)), int(data.get("b", 0)), int(data.get("a", 255)))


def random_roi_color() -> QtGui.QColor:
    return QtGui.QColor(random.randint(40, 255), random.randint(40, 255), random.randint(40, 255))


def timestamp_for_folder() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def timestamp_iso() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------
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

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "type": self.roi_type,
            "points": [[int(x), int(y)] for x, y in self.points],
            "color": qcolor_to_rgba_dict(self.color),
            "area_px": self.area_px,
        }


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
        """Refresh statistics while the dialog is open and camera/playback frames are updating."""
        self.temp_img = temp_img
        self.populate(self.rois, self.temp_img)

    def populate(self, rois: List[Roi], temp_img: Optional[np.ndarray]) -> None:
        self.table.setRowCount(len(rois))
        for row, roi in enumerate(rois):
            stats = compute_roi_statistics(temp_img, roi)
            values = [
                roi.roi_type,
                str(roi.area_px),
                format_stat(stats.get("min_c")),
                format_stat(stats.get("max_c")),
                format_stat(stats.get("mean_c")),
                format_stat(stats.get("std_c")),
            ]
            for col, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table.setItem(row, col, item)


def compute_roi_statistics(temp_img: Optional[np.ndarray], roi: Roi) -> Dict[str, Optional[float]]:
    if temp_img is None or temp_img.shape[:2] != roi.mask.shape:
        return {"min_c": None, "max_c": None, "mean_c": None, "std_c": None}

    roi_values = temp_img[roi.mask > 0]
    if roi_values.size == 0:
        return {"min_c": None, "max_c": None, "mean_c": None, "std_c": None}

    return {
        "min_c": float(np.min(roi_values)),
        "max_c": float(np.max(roi_values)),
        "mean_c": float(np.mean(roi_values)),
        "std_c": float(np.std(roi_values)),
    }


def format_stat(value: Optional[float]) -> str:
    return "—" if value is None else f"{value:.2f}"


# -----------------------------------------------------------------------------
# Main window
# -----------------------------------------------------------------------------
class BARTWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(find_ui_file(), self)

        # ROI drawing state
        self.active_tool: Optional[str] = None
        self.current_points: List[Point] = []
        self.current_hover_point: Optional[Point] = None
        self.rois: List[Roi] = []
        self.statistics_dialogs: List[RoiStatisticsDialog] = []

        # Camera and image state
        self.camera = None
        self.current_temp_img: Optional[np.ndarray] = None
        self.current_display_img: Optional[np.ndarray] = None
        self.image_shape = (320, 320)  # Updated after first frame

        # Recording state
        self.recording_root_dir: Optional[str] = None
        self.recording_session_dir: Optional[str] = None
        self.recording_raw_dir: Optional[str] = None
        self.recording_annotated_dir: Optional[str] = None
        self.recording_stats_path: Optional[str] = None
        self.recording_csv_file = None
        self.recording_csv_writer: Optional[csv.DictWriter] = None
        self.is_recording = False
        self.recorded_frame_index = 0

        # Playback state
        self.is_playback_mode = False
        self.is_playback_running = False
        self.playback_session_dir: Optional[str] = None
        self.playback_raw_files: List[str] = []
        self.playback_index = 0
        self.playback_timer = QtCore.QTimer(self)
        self.playback_timer.timeout.connect(self.advance_playback_frame)
        self.playback_timer.setInterval(100)  # default 10 fps playback

        # Graphics scene
        self.scene = QtWidgets.QGraphicsScene(self)
        self.pixmap_item = QtWidgets.QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.graphicsView.setScene(self.scene)
        self.graphicsView.setMouseTracking(True)
        self.graphicsView.viewport().setMouseTracking(True)
        self.graphicsView.viewport().installEventFilter(self)
        self.graphicsView.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)

        self.configure_buttons()
        self.configure_menu_actions()
        self.configure_roi_table()
        self.configure_recording_playback_controls()
        self.start_camera()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    # ------------------------------------------------------------------ setup
    def configure_buttons(self) -> None:
        self.tool_buttons = {
            "Rectangle": self.rectangleButton,
            "Ellipse": self.ellipseButton,
            "Polygon": self.polygonButton,
        }
        for tool_name, button in self.tool_buttons.items():
            button.setCheckable(True)
            button.clicked.connect(lambda checked, name=tool_name: self.activate_tool(name))

    def configure_menu_actions(self) -> None:
        # Object names are taken from the provided window.ui.
        self.action_Recording_dir.triggered.connect(self.select_recording_directory)
        self.actionSave_current_frame.triggered.connect(self.save_current_frame)
        self.action_Start_recording.triggered.connect(self.start_recording)
        self.actionS_top_recording.triggered.connect(self.stop_recording)
        self.action_Open_recording.triggered.connect(self.open_recording_session)
        self.action_Exit.triggered.connect(self.close)
        self.action_About.triggered.connect(self.show_about_dialog)

        # Settings placeholders: these are connected so the UI does something useful already.
        self.action_Recording_settings.triggered.connect(self.show_recording_settings_placeholder)
        self.action_Temperature_conversion.triggered.connect(self.show_temperature_conversion_placeholder)

    def configure_recording_playback_controls(self) -> None:
        self.snapshotButton.clicked.connect(self.save_current_frame)
        self.startRecordingButton.clicked.connect(self.start_recording)
        self.stopButtonRecording.clicked.connect(self.stop_recording)
        self.openRecordingButton.clicked.connect(self.open_recording_session)
        self.playButton.clicked.connect(self.play_loaded_recording)
        self.pauseButton.clicked.connect(self.pause_playback)
        self.horizontalSlider.valueChanged.connect(self.slider_changed)
        self.horizontalSlider.setEnabled(False)
        self.pauseButton.setEnabled(False)
        self.stopButtonRecording.setEnabled(False)

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

    # ------------------------------------------------------------- tool state
    def activate_tool(self, tool_name: str) -> None:
        """Activate one drawing tool and reset any unfinished shape."""
        self.active_tool = tool_name
        self.current_points.clear()
        self.current_hover_point = None

        for name, button in self.tool_buttons.items():
            is_active = name == tool_name
            button.blockSignals(True)
            button.setChecked(is_active)
            button.setEnabled(not is_active)
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

    # ------------------------------------------------------ camera/rendering
    def update_frame(self) -> None:
        """Main live-camera update loop. Disabled while browsing playback sessions."""
        if self.is_playback_mode:
            return

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

        if self.is_recording:
            self.record_current_frame()

    def simulated_temperature_frame(self) -> np.ndarray:
        h, w = self.image_shape
        y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        x = np.linspace(0, 1, w, dtype=np.float32)[None, :]
        return 22.0 + 8.0 * x + 4.0 * y + np.random.normal(0, 0.08, (h, w)).astype(np.float32)

    def make_annotated_image(self) -> Optional[np.ndarray]:
        """Return the visible image with committed ROIs and active preview drawn on top."""
        if self.current_display_img is None:
            return None
        image = self.current_display_img.copy()
        self.draw_committed_rois(image)
        self.draw_preview(image)
        return image

    def render_scene(self) -> None:
        image = self.make_annotated_image()
        if image is None:
            return

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
            is_closed = len(points) >= 3
            cv2.polylines(image, [np.array(points, dtype=np.int32)], is_closed, color, thickness, cv2.LINE_AA)

    # --------------------------------------------------------------- events
    def eventFilter(self, obj, event):
        if obj is self.graphicsView.viewport():
            if event.type() == QtCore.QEvent.MouseMove:
                point = self.view_to_image_point(event.pos())
                if point is not None and not self.is_playback_mode:
                    self.current_hover_point = point
                    self.render_scene()
                return False

            if event.type() == QtCore.QEvent.MouseButtonPress and event.button() == QtCore.Qt.LeftButton:
                point = self.view_to_image_point(event.pos())
                if point is not None and self.active_tool and not self.is_playback_mode:
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

    # ------------------------------------------------------------ ROI logic
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
        self.save_rois_if_recording()
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

    def refresh_roi_table(self) -> None:
        self.tableWidget.setRowCount(0)
        for roi in self.rois:
            self.add_roi_table_row(roi)

    def delete_selected_rois(self) -> None:
        """Delete selected ROI rows and their corresponding shapes."""
        selected_rows = sorted({index.row() for index in self.tableWidget.selectedIndexes()}, reverse=True)
        if not selected_rows:
            return

        for row in selected_rows:
            if 0 <= row < len(self.rois):
                del self.rois[row]
                self.tableWidget.removeRow(row)

        self.save_rois_if_recording()
        self.render_scene()
        self.update_open_statistics_dialogs()
        self.statusBar().showMessage(f"Deleted {len(selected_rows)} ROI(s).")

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
            self.save_rois_if_recording()
            self.render_scene()

    # ---------------------------------------------------------- statistics
    def open_statistics_dialog(self, *args) -> None:
        dialog = RoiStatisticsDialog(self.rois, self.current_temp_img, self)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        dialog.destroyed.connect(lambda *_: self.update_open_statistics_dialogs())
        self.statistics_dialogs.append(dialog)
        dialog.show()

    def update_open_statistics_dialogs(self) -> None:
        """Refresh all visible statistics dialogs and forget dialogs that were closed."""
        still_open = []
        for dialog in self.statistics_dialogs:
            if dialog.isVisible():
                dialog.update_statistics(self.current_temp_img)
                still_open.append(dialog)
        self.statistics_dialogs = still_open

    # ----------------------------------------------------------- recording
    def select_recording_directory(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select recording directory")
        if directory:
            self.recording_root_dir = directory
            self.statusBar().showMessage(f"Recording directory: {directory}")

    def ensure_recording_root_dir(self) -> bool:
        if self.recording_root_dir and os.path.isdir(self.recording_root_dir):
            return True
        self.select_recording_directory()
        return bool(self.recording_root_dir and os.path.isdir(self.recording_root_dir))

    def create_recording_session(self) -> bool:
        if not self.ensure_recording_root_dir():
            return False

        session_name = f"bart_session_{timestamp_for_folder()}"
        self.recording_session_dir = os.path.join(self.recording_root_dir, session_name)
        self.recording_raw_dir = os.path.join(self.recording_session_dir, "frames_raw")
        self.recording_annotated_dir = os.path.join(self.recording_session_dir, "frames_annotated")
        os.makedirs(self.recording_raw_dir, exist_ok=True)
        os.makedirs(self.recording_annotated_dir, exist_ok=True)

        self.save_session_metadata()
        self.save_rois_json(os.path.join(self.recording_session_dir, "rois.json"))
        self.start_statistics_csv(os.path.join(self.recording_session_dir, "statistics.csv"))
        self.recorded_frame_index = 0
        return True

    def start_recording(self) -> None:
        if self.is_playback_mode:
            QtWidgets.QMessageBox.warning(self, "Recording", "Close playback mode before recording.")
            return
        if self.is_recording:
            return
        if not self.create_recording_session():
            return

        self.is_recording = True
        self.startRecordingButton.setEnabled(False)
        self.stopButtonRecording.setEnabled(True)
        self.statusBar().showMessage(f"Recording started: {self.recording_session_dir}")

    def stop_recording(self) -> None:
        if not self.is_recording:
            return
        self.is_recording = False
        self.close_statistics_csv()
        self.save_session_metadata()
        self.save_rois_if_recording(force=True)
        self.startRecordingButton.setEnabled(True)
        self.stopButtonRecording.setEnabled(False)
        self.statusBar().showMessage(f"Recording stopped. Frames saved: {self.recorded_frame_index}")

    def save_current_frame(self) -> None:
        if self.current_temp_img is None:
            QtWidgets.QMessageBox.warning(self, "Save frame", "No current frame is available yet.")
            return
        if not self.ensure_recording_root_dir():
            return

        frame_dir = os.path.join(self.recording_root_dir, f"single_frame_{timestamp_for_folder()}")
        os.makedirs(frame_dir, exist_ok=True)
        raw_path = os.path.join(frame_dir, "raw_temperature_c.npy")
        png_path = os.path.join(frame_dir, "annotated_frame.png")
        roi_path = os.path.join(frame_dir, "rois.json")
        stats_path = os.path.join(frame_dir, "statistics.csv")

        np.save(raw_path, self.current_temp_img.astype(np.float32))
        annotated = self.make_annotated_image()
        if annotated is not None:
            cv2.imwrite(png_path, annotated)
        self.save_rois_json(roi_path)
        self.write_single_frame_statistics_csv(stats_path)
        self.statusBar().showMessage(f"Saved frame to: {frame_dir}")

    def record_current_frame(self) -> None:
        if self.current_temp_img is None or not self.recording_raw_dir or not self.recording_annotated_dir:
            return

        idx = self.recorded_frame_index
        frame_name = f"frame_{idx:06d}"
        np.save(os.path.join(self.recording_raw_dir, f"{frame_name}.npy"), self.current_temp_img.astype(np.float32))

        annotated = self.make_annotated_image()
        if annotated is not None:
            cv2.imwrite(os.path.join(self.recording_annotated_dir, f"{frame_name}.png"), annotated)

        self.write_statistics_rows(idx, timestamp_iso())
        self.recorded_frame_index += 1

    def start_statistics_csv(self, path: str) -> None:
        self.recording_stats_path = path
        self.recording_csv_file = open(path, "w", newline="", encoding="utf-8")
        fieldnames = [
            "frame_index", "timestamp", "roi_index", "type", "area_px",
            "min_c", "max_c", "mean_c", "std_c",
        ]
        self.recording_csv_writer = csv.DictWriter(self.recording_csv_file, fieldnames=fieldnames)
        self.recording_csv_writer.writeheader()

    def close_statistics_csv(self) -> None:
        if self.recording_csv_file is not None:
            self.recording_csv_file.flush()
            self.recording_csv_file.close()
        self.recording_csv_file = None
        self.recording_csv_writer = None

    def write_statistics_rows(self, frame_index: int, timestamp: str) -> None:
        if self.recording_csv_writer is None:
            return
        for roi_index, roi in enumerate(self.rois):
            stats = compute_roi_statistics(self.current_temp_img, roi)
            self.recording_csv_writer.writerow({
                "frame_index": frame_index,
                "timestamp": timestamp,
                "roi_index": roi_index,
                "type": roi.roi_type,
                "area_px": roi.area_px,
                "min_c": stats.get("min_c"),
                "max_c": stats.get("max_c"),
                "mean_c": stats.get("mean_c"),
                "std_c": stats.get("std_c"),
            })
        if self.recording_csv_file is not None:
            self.recording_csv_file.flush()

    def write_single_frame_statistics_csv(self, path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            fieldnames = ["roi_index", "type", "area_px", "min_c", "max_c", "mean_c", "std_c"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for roi_index, roi in enumerate(self.rois):
                stats = compute_roi_statistics(self.current_temp_img, roi)
                writer.writerow({
                    "roi_index": roi_index,
                    "type": roi.roi_type,
                    "area_px": roi.area_px,
                    "min_c": stats.get("min_c"),
                    "max_c": stats.get("max_c"),
                    "mean_c": stats.get("mean_c"),
                    "std_c": stats.get("std_c"),
                })

    def save_session_metadata(self) -> None:
        if not self.recording_session_dir:
            return
        metadata = {
            "application": "Boson Area Radiometry Tool",
            "created_or_updated": timestamp_iso(),
            "frame_count": self.recorded_frame_index,
            "raw_frame_format": "NumPy .npy, float32 temperature in Celsius",
            "annotated_frame_format": "PNG, 8-bit normalized display image with ROI overlay",
            "temperature_conversion": "Assumed Boson TLinear: Celsius = raw / 100 - 273.15",
            "image_shape": list(self.image_shape),
        }
        with open(os.path.join(self.recording_session_dir, "session.json"), "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    def save_rois_if_recording(self, force: bool = False) -> None:
        if (self.is_recording or force) and self.recording_session_dir:
            self.save_rois_json(os.path.join(self.recording_session_dir, "rois.json"))

    def save_rois_json(self, path: str) -> None:
        data = {
            "image_shape": list(self.image_shape),
            "rois": [roi.to_json_dict() for roi in self.rois],
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    # ------------------------------------------------------------ playback
    def open_recording_session(self) -> None:
        if self.is_recording:
            QtWidgets.QMessageBox.warning(self, "Open recording", "Stop the current recording before opening playback.")
            return

        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Open recording session")
        if not directory:
            return

        raw_dir = os.path.join(directory, "frames_raw")
        if not os.path.isdir(raw_dir):
            QtWidgets.QMessageBox.warning(self, "Open recording", "Selected directory does not contain frames_raw/.")
            return

        raw_files = sorted(
            os.path.join(raw_dir, name)
            for name in os.listdir(raw_dir)
            if name.lower().endswith(".npy")
        )
        if not raw_files:
            QtWidgets.QMessageBox.warning(self, "Open recording", "No .npy frames were found in frames_raw/.")
            return

        self.is_playback_mode = True
        self.is_playback_running = False
        self.playback_session_dir = directory
        self.playback_raw_files = raw_files
        self.playback_index = 0
        self.deactivate_tool()
        self.load_rois_from_session(directory)

        self.horizontalSlider.blockSignals(True)
        self.horizontalSlider.setMinimum(0)
        self.horizontalSlider.setMaximum(len(raw_files) - 1)
        self.horizontalSlider.setValue(0)
        self.horizontalSlider.setEnabled(True)
        self.horizontalSlider.blockSignals(False)

        self.pauseButton.setEnabled(True)
        self.playButton.setEnabled(True)
        self.startRecordingButton.setEnabled(False)
        self.snapshotButton.setEnabled(True)

        self.show_playback_frame(0)
        self.statusBar().showMessage(f"Loaded recording: {directory}")

    def load_rois_from_session(self, directory: str) -> None:
        path = os.path.join(directory, "rois.json")
        if not os.path.exists(path):
            self.rois.clear()
            self.refresh_roi_table()
            return

        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        rois = []
        for item in data.get("rois", []):
            points = [tuple(map(int, p)) for p in item.get("points", [])]
            roi_type = item.get("type", "Unknown")
            color = rgba_dict_to_qcolor(item.get("color", {}))
            mask = self.create_mask_from_points_for_shape(roi_type, points)
            if mask is not None:
                rois.append(Roi(roi_type, points, mask, color))
        self.rois = rois
        self.refresh_roi_table()

    def create_mask_from_points_for_shape(self, roi_type: str, points: List[Point]) -> Optional[np.ndarray]:
        return self.create_mask(roi_type, points)

    def play_loaded_recording(self) -> None:
        if not self.playback_raw_files:
            self.open_recording_session()
            return
        self.is_playback_mode = True
        self.is_playback_running = True
        self.playback_timer.start()
        self.statusBar().showMessage("Playback started.")

    def pause_playback(self) -> None:
        self.is_playback_running = False
        self.playback_timer.stop()
        self.statusBar().showMessage("Playback paused.")

    def advance_playback_frame(self) -> None:
        if not self.playback_raw_files:
            self.pause_playback()
            return
        next_index = self.playback_index + 1
        if next_index >= len(self.playback_raw_files):
            next_index = 0
        self.show_playback_frame(next_index)

    def slider_changed(self, value: int) -> None:
        if self.is_playback_mode and self.playback_raw_files:
            self.show_playback_frame(value)

    def show_playback_frame(self, index: int) -> None:
        if not self.playback_raw_files:
            return
        index = int(np.clip(index, 0, len(self.playback_raw_files) - 1))
        self.playback_index = index
        self.current_temp_img = np.load(self.playback_raw_files[index]).astype(np.float32)
        self.image_shape = self.current_temp_img.shape[:2]
        display_gray = normalize_for_display(self.current_temp_img)
        self.current_display_img = cv2.cvtColor(display_gray, cv2.COLOR_GRAY2BGR)

        self.horizontalSlider.blockSignals(True)
        self.horizontalSlider.setValue(index)
        self.horizontalSlider.blockSignals(False)

        self.render_scene()
        self.update_open_statistics_dialogs()
        self.statusBar().showMessage(f"Playback frame {index + 1}/{len(self.playback_raw_files)}")

    def leave_playback_mode(self) -> None:
        self.pause_playback()
        self.is_playback_mode = False
        self.playback_raw_files.clear()
        self.horizontalSlider.setEnabled(False)
        self.startRecordingButton.setEnabled(True)
        self.statusBar().showMessage("Returned to live camera mode.")

    # -------------------------------------------------------------- dialogs
    def show_about_dialog(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "About BART",
            "Boson Area Radiometry Tool\n\n"
            "Live ROI temperature measurement and recording tool for FLIR Boson cameras.",
        )

    def show_recording_settings_placeholder(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "Recording settings",
            "Recording settings can be added here later, e.g. FPS limit, save raw frames, save annotated frames, and CSV export options.",
        )

    def show_temperature_conversion_placeholder(self) -> None:
        QtWidgets.QMessageBox.information(
            self,
            "Temperature conversion",
            "Current assumption:\n\nTemperature [°C] = raw / 100 - 273.15\n\nAdd emissivity/reflected-temperature settings here later if needed.",
        )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.is_recording:
            self.stop_recording()
        self.pause_playback()
        if self.camera is not None:
            try:
                self.camera.close()
            except Exception:
                pass
        event.accept()


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------
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
