
import sys
import csv
from PySide6.QtCore import Qt
from PySide6.QtGui import QPen, QBrush, QColor, QAction, QPixmap, QPainter, QFont
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene,
    QGraphicsEllipseItem, QGraphicsTextItem, QGraphicsRectItem, QGraphicsPixmapItem
)

# Room dimensions and Y-axis quantization
R_Width, R_Height = 690, 830
W_Width, W_Height = 6.9, 8.3
Y_VALUES = [1.5, 4.0, 6.5]

# Y-axis quantization function
def quantize_y(y):
    return min(Y_VALUES, key=lambda v: abs(y - v))

# Convert world coordinates to scene coordinates
def world_to_scene(x, y):
    sx = (x / W_Width) * R_Width
    sy = (y / W_Height) * R_Height
    return sx, sy

# Stops visualization viewer
class StopsViewerYQuantized(QMainWindow):
    def __init__(self, txt_files):
        super().__init__()
        # Window setup
        self.setWindowTitle("UI (Y=1.5,4,6.5)")
        self.resize(R_Width + 100, R_Height + 50)
        
        # Central view
        self.scene.setSceneRect(0, 0, R_Width, R_Height)
        self.view = QGraphicsView(self.scene, self)
        self.setCentralWidget(self.view)
        
        self._configure_view()

        # Initialize room visualization
        self._draw_y_lines()
        self._draw_gray_zones()
        
        # Load all files
        if isinstance(txt_files, str):
            txt_files = [txt_files]
        for txt_file in txt_files:
            self._load_and_display_stops(txt_file)
        
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
            bg = QGraphicsRectItem(0, 0, R_Width, R_Height)
            bg.setBrush(QBrush(QColor(255, 255, 255)))
            bg.setPen(QPen(Qt.NoPen))
            bg.setZValue(0)
            self.scene.addItem(bg)
        
        border_pen = QPen(QColor(0, 0, 0), 2)
        border = QGraphicsRectItem(0, 0, R_Width, R_Height)
        border.setPen(border_pen)
        border.setBrush(QBrush(Qt.NoBrush))
        border.setZValue(1)
        self.scene.addItem(border)
    
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
        
        for x_m, y_m, w_m, h_m in zones:
            sx, sy = world_to_scene(x_m, y_m)
            sw = (w_m / W_Width) * R_Width
            sh = (h_m / W_Height) * R_Height
            
            zone = QGraphicsRectItem(sx, sy, sw, sh)
            zone.setBrush(QBrush(QColor(120, 120, 120, 100)))
            zone.setPen(QPen(Qt.NoPen))
            zone.setZValue(0.6)
            self.scene.addItem(zone)
    
    def _load_and_display_stops(self, txt_file):
        # Read stops file and display on visualization
        with open(txt_file, 'r', encoding='utf-8') as f:
            header = f.readline()
            stop_number = 1
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(',')
                stop_id = int(parts[0])
                x = float(parts[1])
                y_raw = float(parts[2])
                
                y = quantize_y(y_raw)
                
                sx, sy = world_to_scene(x, y)
                
                radius = 8
                dot = QGraphicsEllipseItem(
                    sx - radius, sy - radius,
                    2 * radius, 2 * radius
                )
                dot.setBrush(QBrush(QColor(9, 203, 15, 200)))
                dot.setPen(QPen(QColor(0, 150, 0), 2))
                dot.setZValue(10)
                self.scene.addItem(dot)
                
                stop_number += 1

def main():
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('txt_files', nargs='+')
    
    args = parser.parse_args()
    
    # Create and display viewer
    app = QApplication(sys.argv)
    viewer = StopsViewerYQuantized(args.txt_files)
    viewer.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
