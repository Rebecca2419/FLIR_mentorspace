

import sys
import argparse
from datetime import datetime
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPen, QBrush, QColor, QAction, QPixmap, QPainter
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QGraphicsEllipseItem, QGraphicsTextItem, QDockWidget,
    QListWidget, QTableWidget, QTableWidgetItem, QToolBar, QGraphicsPixmapItem
)
from connect import get_connection

R_Width, R_Height = 690, 830
W_Width, W_Height = 6.9, 8.3
X_GRID_SIZE = 0.3
Y_VALUES = [1.5, 4.0, 6.5]

def quantize_y(y):
    return min(Y_VALUES, key=lambda v: abs(y - v))

def world_to_quantized(x, y):
    grid_x = int(x / X_GRID_SIZE)
    y_value = quantize_y(y)
    return grid_x, y_value

def quantized_to_world(grid_x, y_value):
    x = (grid_x + 0.5) * X_GRID_SIZE
    y = y_value
    return x, y

def world_to_scene(x, y):
    sx = (x / W_Width) * R_Width
    sy = (y / W_Height) * R_Height
    return sx, sy

class AppMainWin(QMainWindow):
    def __init__(self, timestamp=None):
        super().__init__()
        # Window setup
        self.setWindowTitle("People UI - Y-Quantized Mode (Y=1.5,4,6.5)")
        self.resize(R_Width + 100, R_Height + 10)
        
        # State initialization
        self.timestamp = timestamp
        self.recordings = []
        self.current_recording = None
        self.recording_data = {}
        self.current_frame_index = 0
        self.people = {}  # Track person positions and graphics
        self.trails = []  # Track movement trails
        self.is_playing = False
        
        # Central view
        self.scene = QGraphicsScene(self)
        self.scene.setSceneRect(0, 0, R_Width, R_Height)
        self.view = QGraphicsView(self.scene, self)
        self.setCentralWidget(self.view)
        self._configure_view()
        self._init_room()
        self._draw_x_grid()
        self._draw_y_lines()
        self._draw_gray_zones()
        

        self.recordingsDock = QDockWidget("Saved classes", self)
        self.recordingsList = QListWidget()
        self.recordingsList.setMinimumWidth(240)
        font = self.recordingsList.font()
        font.setPointSize(12)
        self.recordingsList.setFont(font)
        self.recordingsList.setSpacing(8)
        self.recordingsDock.setWidget(self.recordingsList)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.recordingsDock)
        
        self.peopleDock = QDockWidget("People Info", self)
        self.peopleTable = QTableWidget(0, 2)
        self.peopleTable.setHorizontalHeaderLabels(["X", "Y"])
        self.peopleTable.setColumnWidth(0, 120)
        self.peopleTable.setColumnWidth(1, 120)
        self.peopleTable.setRowHeight(0, 40)
        font = self.peopleTable.font()
        font.setPointSize(14)
        self.peopleTable.setFont(font)
        self.peopleDock.setWidget(self.peopleTable)
        self.addDockWidget(Qt.RightDockWidgetArea, self.peopleDock)
        
        tb = QToolBar("Controls", self)
        self.addToolBar(tb)
        self.actionPlay = QAction("Play", self)
        self.actionPause = QAction("Pause", self)
        self.actionRefresh = QAction("Refresh", self)
        tb.addAction(self.actionPlay)
        tb.addAction(self.actionPause)
        tb.addSeparator()
        tb.addAction(self.actionRefresh)
        self.actionPlay.triggered.connect(self.start_feed)
        self.actionPause.triggered.connect(self.stop_feed)
        self.actionRefresh.triggered.connect(self._refresh_scene)
        
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self._tick_playback)
        
        self.db = get_connection()
        self.recordingsList.addItem("Loading...")
        QTimer.singleShot(100, self._load_recordings_from_database)
        
        self._fit_scene()
        
    def _configure_view(self):
        # Configure view rendering and interaction
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.view.setRenderHints(self.view.renderHints() | QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self._auto_fit = True
        
    def _fit_scene(self):
        # Fit scene to view with aspect ratio
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
        
    def resizeEvent(self, event):
        # Handle window resize with auto-fit
        super().resizeEvent(event)
        if getattr(self, "_auto_fit", False):
            self._fit_scene()
            
    def _init_room(self):
        # Load UI scene PNG as background
        pixmap = QPixmap("ui sc.png")
        if not pixmap.isNull():
            # Scale pixmap to match room dimensions
            pixmap = pixmap.scaledToWidth(R_Width, Qt.SmoothTransformation)
            bg_item = QGraphicsPixmapItem(pixmap)
            bg_item.setPos(0, 0)
            bg_item.setZValue(0)
            self.scene.addItem(bg_item)
        else:
            # Fallback to white background if image not found
            background = self.scene.addRect(0, 0, R_Width, R_Height)
            background.setBrush(QBrush(QColor(255, 255, 255)))
            background.setPen(QPen(QColor(0, 0, 0), 2))
            background.setZValue(0)
    
    def _draw_x_grid(self):
        # Draw X-axis grid lines at 0.3m spacing
        pen = QPen(QColor(200, 200, 200, 100))
        pen.setStyle(Qt.DashLine)
        pen.setWidth(1)
        
        x = 0
        while x <= W_Width:
            sx, _ = world_to_scene(x, 0)
            _, sy_bottom = world_to_scene(x, W_Height)
            line = self.scene.addLine(sx, 0, sx, sy_bottom, pen)
            line.setZValue(0.5)
            x += X_GRID_SIZE

    def _draw_y_lines(self):
        # Draw Y-axis quantization lines at 1.5, 4.0, 6.5
        pen = QPen(QColor(150, 150, 200, 150))
        pen.setStyle(Qt.DashLine)
        pen.setWidth(1)
        
        for y_val in Y_VALUES:
            _, sy = world_to_scene(0, y_val)
            sx_right, _ = world_to_scene(W_Width, y_val)
            line = self.scene.addLine(0, sy, sx_right, sy, pen)
            line.setZValue(0.5)

    def _draw_gray_zones(self):
        # Mark restricted zones
        zones = [
            (1.55, 2.12, 3.6, 1.2),
            (1.55, 4.58, 3.6, 1.2),
        ]
        brush = QBrush(QColor(120, 120, 120, 90))
        pen = QPen(Qt.NoPen)
        for x, y, w, h in zones:
            sx, sy = world_to_scene(x, y)
            sw = (w / W_Width) * R_Width
            sh = (h / W_Height) * R_Height
            rect = self.scene.addRect(sx, sy, sw, sh, pen, brush)
            rect.setZValue(0.6)
            
    def _load_recordings_from_database(self):
        # Fetch and display all recording sessions
        if not self.db:
            self.recordingsList.clear()
            self.recordingsList.addItem("Connection Failed")
            return
        try:
            tables = self.db.get_tables()
            sessions = {}
            
            for table in tables:
                if isinstance(table, str) and table.startswith("flir_data_"):
                    date_str = table.split("_")[-1]
                    try:
                        date_obj = datetime.strptime(date_str, "%Y%m%d")
                        sessions[date_obj] = table
                    except ValueError:
                        continue

            sorted_dates = sorted(sessions.keys(), reverse=True)
            self.recordings = []
            self.recordingsList.clear()
            
            for date_obj in sorted_dates:
                table = sessions[date_obj]
                display_text = f"Session {date_obj.strftime('%Y-%m-%d')}"
                self.recordingsList.addItem(display_text)
                self.recordings.append({
                    "date": date_obj,
                    "table": table,
                    "display": display_text
                })
                
            if self.recordings:
                self.recordingsList.setCurrentRow(0)
                self.recordingsList.itemClicked.connect(self._on_recording_selected)
                
        except Exception as e:
            self.recordingsList.clear()
            self.recordingsList.addItem(f"Error: {str(e)}")
            
    def _on_recording_selected(self, item):
        # Load selected recording session
        index = self.recordingsList.row(item)
        if 0 <= index < len(self.recordings):
            self.current_recording = self.recordings[index]
            self._load_recording_data()
            self.current_frame_index = 0
            self._display_frame()
            
    def _load_recording_data(self):
        # Query database for recording data
        if not self.current_recording or not self.db:
            return
        try:
            table = self.current_recording["table"]
            query = f"""
                SELECT timestamp, data_id, x_axis, y_axis, valid
                FROM {table}
                ORDER BY timestamp, data_id
            """
            results = self.db.execute_query(query)
            
            self.recording_data = {}
            for row in results:
                ts = row["timestamp"]
                if ts not in self.recording_data:
                    self.recording_data[ts] = []
                self.recording_data[ts].append({
                    "data_id": row["data_id"],
                    "x": row["x_axis"],
                    "y": row["y_axis"],
                    "valid": row.get("valid", True)
                })
        except Exception as e:
            self.recordingsList.clear()
            self.recordingsList.addItem(f"Error: {str(e)}")
            
    def _display_frame(self):
        # Render current frame with all people positions
        if not self.recording_data:
            return
        timestamps = sorted(self.recording_data.keys())
        if not timestamps:
            return
        if self.current_frame_index >= len(timestamps):
            self.current_frame_index = 0
            
        current_ts = timestamps[self.current_frame_index]
        frame_data = self.recording_data[current_ts]
        
        valid_person_ids = set()
        
        for person_info in frame_data:
            person_id = person_info["data_id"]
            x = person_info["x"]
            y = person_info["y"]
            valid = person_info.get("valid", True)
            
            if valid:
                valid_person_ids.add(person_id)
                self.handle_update(person_id, x, y)
            else:
                if person_id in self.people:
                    entry = self.people[person_id]
                    entry["dot"].setVisible(False)
                    entry["label"].setVisible(False)
                    entry["last_pos"] = None
        
        for person_id in valid_person_ids:
            if person_id in self.people:
                entry = self.people[person_id]
                entry["dot"].setVisible(True)
                entry["label"].setVisible(True)
            
        self._update_people_table()
            
    def start_feed(self):
        # Start playback
        self.is_playing = True
        self.timer.start()
        
    def stop_feed(self):
        # Stop playback
        self.is_playing = False
        self.timer.stop()
        
    def handle_update(self, person_id: int, x: float, y: float):
        # Update person position and trail visualization
        grid_x, y_value = world_to_quantized(x, y)
        
        center_x, center_y = quantized_to_world(grid_x, y_value)
        
        sx, sy = world_to_scene(center_x, center_y)
        
        r = 8
        
        if person_id not in self.people:
            dot = QGraphicsEllipseItem(0, 0, 2 * r, 2 * r)
            dot.setBrush(QBrush(QColor("#09cb0f")))
            dot.setPen(QPen(Qt.NoPen))
            dot.setZValue(10)
            
            label = QGraphicsTextItem(str(person_id))
            label.setDefaultTextColor(QColor("#1a1a1a"))
            label.setZValue(11)
            
            self.scene.addItem(dot)
            self.scene.addItem(label)
            
            self.people[person_id] = {
                "dot": dot,
                "label": label,
                "last_pos": None
            }
        
        entry = self.people[person_id]
        dot = entry["dot"]
        label = entry["label"]
        
        last_pos = entry["last_pos"]
        current_pos = (grid_x, y_value)
        
        if last_pos is not None and last_pos != current_pos:
            last_center_x, last_center_y = quantized_to_world(last_pos[0], last_pos[1])
            lx, ly = world_to_scene(last_center_x, last_center_y)
            seg = self.scene.addLine(lx, ly, sx, sy, QPen(QColor(180, 180, 180)))
            seg.setZValue(1)
            self.trails.append(seg)
        
        dot.setPos(sx - r, sy - r)
        label.setPos(sx + r + 2, sy - r)
        
        entry["last_pos"] = current_pos
        
    def _refresh_scene(self):
        # Clear all visualization and reset state
        self.is_playing = False
        self.timer.stop()
        
        for person_id in list(self.people.keys()):
            entry = self.people[person_id]
            self.scene.removeItem(entry["dot"])
            self.scene.removeItem(entry["label"])
        self.people.clear()
        
        for line in self.trails:
            self.scene.removeItem(line)
        self.trails.clear()
        
        self.current_frame_index = 0
        self.peopleTable.setRowCount(0)
        
    def _update_people_table(self):
        # Update people info display table
        if not hasattr(self, 'peopleTable'):
            return
            
        ids = sorted(self.people.keys())
        self.peopleTable.setRowCount(len(ids))
        
        for row, pid in enumerate(ids):
            entry = self.people[pid]
            
            if entry["last_pos"]:
                grid_x, y_value = entry["last_pos"]
                wx, wy = quantized_to_world(grid_x, y_value)
            else:
                wx, wy = 0.0, 0.0
            
            self.peopleTable.setItem(row, 0, QTableWidgetItem(f"{wx:.2f}"))
            self.peopleTable.setItem(row, 1, QTableWidgetItem(f"{wy:.2f}"))
            self.peopleTable.setRowHeight(row, 40)
            
        self.peopleTable.resizeColumnsToContents()
        
    def _tick_playback(self):
        # Advance to next frame during playback
        if not self.is_playing or not self.recording_data:
            return
            
        self.current_frame_index += 1
        timestamps = sorted(self.recording_data.keys())
        
        if self.current_frame_index >= len(timestamps):
            self.stop_feed()
            return
            
        self._display_frame()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timestamp", type=str)
    parser.add_argument("--date", type=str)
    
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    w = AppMainWin(timestamp=args.timestamp)
    w.show()
    sys.exit(app.exec())
    
if __name__ == "__main__":
    main()
