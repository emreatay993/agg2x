import sys
import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QTableWidget, 
                             QTableWidgetItem, QHeaderView, QSplitter, QMessageBox, 
                             QSpinBox, QDoubleSpinBox, QFrame, QLabel, QMenu, QAction)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPalette
import pyvista as pv
from pyvistaqt import QtInteractor

# --- Custom Styling for "Modern/Minimalistic" Look ---
def apply_modern_style(app):
    """Applies a dark, flat modern theme to the PyQt application."""
    app.setStyle("Fusion")
    
    dark_palette = QPalette()
    dark_bg = QColor(45, 45, 45)
    dark_lighter = QColor(55, 55, 55)
    text_color = QColor(220, 220, 220)
    accent_color = QColor(42, 130, 218) # Modern Blue
    
    dark_palette.setColor(QPalette.Window, dark_bg)
    dark_palette.setColor(QPalette.WindowText, text_color)
    dark_palette.setColor(QPalette.Base, dark_lighter)
    dark_palette.setColor(QPalette.AlternateBase, dark_bg)
    dark_palette.setColor(QPalette.ToolTipBase, text_color)
    dark_palette.setColor(QPalette.ToolTipText, text_color)
    dark_palette.setColor(QPalette.Text, text_color)
    dark_palette.setColor(QPalette.Button, dark_bg)
    dark_palette.setColor(QPalette.ButtonText, text_color)
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, accent_color)
    dark_palette.setColor(QPalette.Highlight, accent_color)
    dark_palette.setColor(QPalette.HighlightedText, Qt.white)
    
    app.setPalette(dark_palette)
    
    # CSS for specific widgets to refine the look
    app.setStyleSheet("""
        QTableWidget {
            gridline-color: #3d3d3d;
            border: none;
        }
        QHeaderView::section {
            background-color: #353535;
            padding: 5px;
            border: 1px solid #3d3d3d;
        }
        QPushButton {
            background-color: #3d3d3d;
            border: 1px solid #555;
            padding: 6px;
            border-radius: 4px;
        }
        QPushButton:hover {
            background-color: #4d4d4d;
            border: 1px solid #2a82da;
        }
        QPushButton:pressed {
            background-color: #2a82da;
        }
        QSplitter::handle {
            background-color: #3d3d3d;
        }
    """)

# --- Data Model ---
class SectorData:
    """
    Handles loading, validation, and geometry generation for a single .dat file.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.filename = os.path.basename(file_path)
        self.df = None
        self.xyz_data = None # Numpy array N x 4
        self.valid = False
        self.error_msg = ""
        
        # Defaults
        self.num_sectors = 1
        self.total_angle = 360.0
        
        self._load_and_validate()

    def _load_and_validate(self):
        try:
            # Attempt to read file. Assume whitespace or csv.
            # Reading header=None because user said "no headers"
            self.df = pd.read_csv(self.file_path, sep=None, engine='python', header=None)
            
            # Validation: Check columns
            if self.df.shape[1] < 4:
                self.valid = False
                self.error_msg = f"Insufficient columns. Found {self.df.shape[1]}, expected at least 4 (X,Y,Z,Data)."
                return

            # Ensure numeric
            cols_to_check = [0, 1, 2, 3]
            self.df[cols_to_check] = self.df[cols_to_check].apply(pd.to_numeric, errors='coerce')
            
            if self.df[cols_to_check].isnull().values.any():
                self.valid = False
                self.error_msg = "Non-numeric data found in the first 4 columns."
                return

            # Store as numpy for speed
            self.xyz_data = self.df.iloc[:, :4].to_numpy()
            self.valid = True

        except Exception as e:
            self.valid = False
            self.error_msg = str(e)

    def get_expanded_mesh(self):
        """
        Generates the PyVista PolyData.
         duplicates the data 'num_sectors' times around Z-axis.
        """
        if not self.valid:
            return None

        # Base data
        base_xyz = self.xyz_data[:, :3]
        scalars = self.xyz_data[:, 3]

        if self.num_sectors == 1:
            # No duplication
            cloud = pv.PolyData(base_xyz)
            cloud["Data"] = scalars
            return cloud

        # Calculate rotation angles
        # If total angle is 360, we usually divide by N (0, 90, 180, 270). 
        # If user puts 180 total for 2 sectors, it implies 0 and 90? 
        # We will assume: Step = TotalAngle / NumSectors
        # If 360 and 4 sectors -> 0, 90, 180, 270.
        
        step = self.total_angle / self.num_sectors
        angles = np.radians([i * step for i in range(self.num_sectors)])
        
        all_points = []
        all_scalars = []

        # Vectorized Rotation:
        # x' = x*cos - y*sin
        # y' = x*sin + y*cos
        x = base_xyz[:, 0]
        y = base_xyz[:, 1]
        z = base_xyz[:, 2]

        for theta in angles:
            c, s = np.cos(theta), np.sin(theta)
            x_rot = x * c - y * s
            y_rot = x * s + y * c
            
            # Reconstruct points for this sector
            sector_pts = np.column_stack((x_rot, y_rot, z))
            all_points.append(sector_pts)
            all_scalars.append(scalars) # Scalars don't change

        # Stack everything
        final_pts = np.vstack(all_points)
        final_scalars = np.hstack(all_scalars)

        cloud = pv.PolyData(final_pts)
        cloud["Data"] = final_scalars
        return cloud
    
    def get_csv_data(self):
        """Regenerates raw data for export."""
        mesh = self.get_expanded_mesh()
        if mesh:
            arr = np.column_stack((mesh.points, mesh["Data"]))
            df = pd.DataFrame(arr, columns=["X", "Y", "Z", "Data"])
            # Add a source column for clarity (optional, but good for export)
            df["SourceFile"] = self.filename
            return df
        return pd.DataFrame()

# --- Main GUI ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sector Data Processor")
        self.resize(1200, 800)
        
        # State
        self.loaded_files = {} # Dict: path -> SectorData
        self.actors = {} # Dict: path -> pyvista_actor
        
        self.setup_ui()

    def setup_ui(self):
        # Main Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Splitter (Left: Controls, Right: Vis)
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # --- LEFT PANEL ---
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        
        # Header
        lbl_title = QLabel("Data Inputs")
        lbl_title.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 5px;")
        left_layout.addWidget(lbl_title)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("Add Files")
        self.btn_add.clicked.connect(self.add_files)
        self.btn_remove = QPushButton("Remove Selected")
        self.btn_remove.clicked.connect(self.remove_file)
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        left_layout.addLayout(btn_layout)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Filename", "Sectors", "Angle (Deg)"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.itemSelectionChanged.connect(self.on_selection_change)
        left_layout.addWidget(self.table)

        # Export Button
        self.btn_export = QPushButton("Export Expanded Data to CSV")
        self.btn_export.setStyleSheet("background-color: #2a82da; color: white; font-weight: bold; padding: 10px;")
        self.btn_export.clicked.connect(self.export_data)
        left_layout.addWidget(self.btn_export)

        # --- RIGHT PANEL (PyVista) ---
        right_panel = QFrame()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # PyVistaQt Widget
        self.plotter = QtInteractor(right_panel)
        self.plotter.set_background("#252525") # Matches dark theme
        self.plotter.add_axes()
        right_layout.addWidget(self.plotter.interactor)
        
        # Custom Context Menu for Plotter
        self.plotter.add_custom_menu_actions() # Default PyVista actions
        
        # We need to hook into the right click to add our "Show All"
        # PyVistaQt doesn't expose a clean signal for right-click menu modification easily
        # so we will use a Qt context menu strategy on the widget itself if needed,
        # but simpler is to use a button or key binding. 
        # However, let's try to override the context menu policy.
        self.plotter.interactor.setContextMenuPolicy(Qt.CustomContextMenu)
        self.plotter.interactor.customContextMenuRequested.connect(self.show_context_menu)

        # Add panels to splitter
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800]) # Initial ratio

    # --- Logic: File Handling ---
    def add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Open Data Files", "", "Dat Files (*.dat)")
        if not paths:
            return

        errors = []
        for path in paths:
            if path in self.loaded_files:
                continue # Skip duplicates
            
            data_obj = SectorData(path)
            if data_obj.valid:
                self.loaded_files[path] = data_obj
                self.add_table_row(data_obj)
            else:
                errors.append(f"{os.path.basename(path)}: {data_obj.error_msg}")
        
        self.update_visualization()

        if errors:
            err_str = "\n".join(errors)
            QMessageBox.warning(self, "Skipped Invalid Files", f"The following files were skipped:\n\n{err_str}")

    def add_table_row(self, data_obj):
        row = self.table.rowCount()
        self.table.insertRow(row)

        # Col 0: Filename (Store path in data)
        item_name = QTableWidgetItem(data_obj.filename)
        item_name.setData(Qt.UserRole, data_obj.file_path)
        item_name.setFlags(item_name.flags() ^ Qt.ItemIsEditable) # Read only
        self.table.setItem(row, 0, item_name)

        # Col 1: Sectors (SpinBox)
        sb_sectors = QSpinBox()
        sb_sectors.setRange(1, 360)
        sb_sectors.setValue(data_obj.num_sectors)
        sb_sectors.valueChanged.connect(lambda val, path=data_obj.file_path: self.on_param_change(path, "sectors", val))
        self.table.setCellWidget(row, 1, sb_sectors)

        # Col 2: Angle (DoubleSpinBox)
        sb_angle = QDoubleSpinBox()
        sb_angle.setRange(1.0, 360.0)
        sb_angle.setValue(data_obj.total_angle)
        sb_angle.setSuffix("°")
        sb_angle.valueChanged.connect(lambda val, path=data_obj.file_path: self.on_param_change(path, "angle", val))
        self.table.setCellWidget(row, 2, sb_angle)

    def remove_file(self):
        current_row = self.table.currentRow()
        if current_row < 0:
            return
        
        item = self.table.item(current_row, 0)
        path = item.data(Qt.UserRole)
        
        # Cleanup
        if path in self.loaded_files:
            del self.loaded_files[path]
        if path in self.actors:
            self.plotter.remove_actor(self.actors[path])
            del self.actors[path]
            
        self.table.removeRow(current_row)
        self.update_visualization_ranges() # Re-calc global limits

    # --- Logic: Updates & Visualization ---
    def on_param_change(self, path, param_type, value):
        if path not in self.loaded_files:
            return
        
        # Update Model
        if param_type == "sectors":
            self.loaded_files[path].num_sectors = value
        elif param_type == "angle":
            self.loaded_files[path].total_angle = value
        
        # Update Visuals
        self.update_single_mesh(path)
        self.update_visualization_ranges()

    def update_single_mesh(self, path):
        """Regenerates mesh for specific file and updates actor."""
        data_obj = self.loaded_files[path]
        mesh = data_obj.get_expanded_mesh()
        
        if path in self.actors:
            self.plotter.remove_actor(self.actors[path])
        
        # Add mesh, keep reference. 
        # Note: We don't set scalar range here immediately, that's done in update_ranges
        actor = self.plotter.add_mesh(mesh, scalars="Data", name=path, show_scalar_bar=False, cmap="viridis")
        self.actors[path] = actor

    def update_visualization(self):
        """Initial load or full refresh."""
        self.plotter.clear()
        self.actors = {}
        
        for path, data_obj in self.loaded_files.items():
            self.update_single_mesh(path)
            
        self.update_visualization_ranges()
        self.plotter.reset_camera()

    def update_visualization_ranges(self):
        """Calculates Min/Max and updates colors based on selection state."""
        if not self.loaded_files:
            return

        # Check selection state
        selected_rows = self.table.selectionModel().selectedRows()
        selected_path = None
        
        if selected_rows:
            row = selected_rows[0].row()
            selected_path = self.table.item(row, 0).data(Qt.UserRole)

        # 1. Determine Min/Max
        if selected_path:
            # Local Min/Max
            mesh = self.actors[selected_path].mapper.dataset
            dmin, dmax = mesh.get_data_range("Data")
        else:
            # Global Min/Max
            g_min, g_max = float('inf'), float('-inf')
            for path in self.loaded_files:
                if path in self.actors:
                    mesh = self.actors[path].mapper.dataset
                    mn, mx = mesh.get_data_range("Data")
                    if mn < g_min: g_min = mn
                    if mx > g_max: g_max = mx
            dmin, dmax = g_min, g_max

        # 2. Update Actors
        for path, actor in self.actors.items():
            if selected_path is None:
                # GLOBAL VIEW: All colored
                actor.mapper.scalar_range = (dmin, dmax)
                actor.prop.opacity = 1.0
                actor.prop.color = None # Reset potential grey override
                actor.mapper.scalar_visibility = True
            else:
                # ISOLATED VIEW
                if path == selected_path:
                    # Highlighted
                    actor.mapper.scalar_range = (dmin, dmax)
                    actor.prop.opacity = 1.0
                    actor.mapper.scalar_visibility = True
                else:
                    # Ghosted
                    actor.mapper.scalar_visibility = False
                    actor.prop.color = "lightgrey"
                    actor.prop.opacity = 0.15

        # Update Scalar Bar
        self.plotter.clear_scalar_bars()
        self.plotter.add_scalar_bar(title="Data Value", limits=(dmin, dmax))

    def on_selection_change(self):
        self.update_visualization_ranges()

    def show_context_menu(self, pos):
        """Right click menu to reset view."""
        menu = QMenu(self)
        reset_action = QAction("Show All (Reset Selection)", self)
        reset_action.triggered.connect(self.reset_selection)
        menu.addAction(reset_action)
        menu.exec_(self.plotter.interactor.mapToGlobal(pos))

    def reset_selection(self):
        self.table.clearSelection()
        # on_selection_change triggers automatically via signal or we call update manually
        # Note: clearSelection triggers selectionChanged with empty selection.

    # --- Logic: Export ---
    def export_data(self):
        if not self.loaded_files:
            return

        file_path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "", "CSV Files (*.csv)")
        if not file_path:
            return

        try:
            dfs = []
            for path, data_obj in self.loaded_files.items():
                dfs.append(data_obj.get_csv_data())
            
            final_df = pd.concat(dfs, ignore_index=True)
            final_df.to_csv(file_path, index=False)
            QMessageBox.information(self, "Success", "Data exported successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export: {str(e)}")

# --- Entry Point ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_modern_style(app)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())
