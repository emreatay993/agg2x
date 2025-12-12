"""
PyMechanical Script: Apply Inertia Relief Accelerations to Static Structural Analysis

This script reads the CSV output from `ansys_inertia_relief_out_to_csv.py` and creates:
  1. A local coordinate system at the Center of Mass (COM)
  2. Translational Acceleration boundary condition (tabular: Time vs Ax, Ay, Az)
  3. Rotational Acceleration boundary condition (tabular: Time vs αx, αy, αz)

Features:
  - PyQt5 GUI for CSV selection and script generation
  - Generates ACT scripts for pasting into Mechanical console
  - Can run directly via PyMechanical (embedded or remote session)
  - Displays project tree on success
  - Saves and closes Mechanical session cleanly

Requirements:
  - Ansys Mechanical 2023 R2 or later
  - An existing Static Structural analysis with geometry/mesh already set up
  - The IR summary CSV file (e.g., ansys_IR_input_summary_solve.csv)

Author: Generated for workflow automation
"""

from __future__ import annotations

import csv
import io
import os
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple


# =============================================================================
# Data Classes
# =============================================================================
@dataclass
class IRLoadStep:
    """Represents one load step of inertia relief data."""
    load_step: int
    time_s: float
    com_x_mm: float
    com_y_mm: float
    com_z_mm: float
    trans_accel_x: float  # m/s² from CSV
    trans_accel_y: float
    trans_accel_z: float
    rot_accel_x: float    # rad/s² from CSV
    rot_accel_y: float
    rot_accel_z: float


def read_ir_summary_csv(csv_path: str) -> Tuple[List[IRLoadStep], Tuple[float, float, float]]:
    """
    Read the IR summary CSV and return load step data + average COM position.
    
    Returns:
        Tuple of (list of IRLoadStep, average COM as (x, y, z) in mm)
    """
    steps: List[IRLoadStep] = []
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = IRLoadStep(
                load_step=int(row["load_step"]),
                time_s=float(row["time_end_s"]),
                com_x_mm=float(row["com_x_mm"]) if row.get("com_x_mm") else 0.0,
                com_y_mm=float(row["com_y_mm"]) if row.get("com_y_mm") else 0.0,
                com_z_mm=float(row["com_z_mm"]) if row.get("com_z_mm") else 0.0,
                trans_accel_x=float(row["trans_accel_x_m_s2"]) if row.get("trans_accel_x_m_s2") else 0.0,
                trans_accel_y=float(row["trans_accel_y_m_s2"]) if row.get("trans_accel_y_m_s2") else 0.0,
                trans_accel_z=float(row["trans_accel_z_m_s2"]) if row.get("trans_accel_z_m_s2") else 0.0,
                rot_accel_x=float(row["rot_accel_x_rad_s2"]) if row.get("rot_accel_x_rad_s2") else 0.0,
                rot_accel_y=float(row["rot_accel_y_rad_s2"]) if row.get("rot_accel_y_rad_s2") else 0.0,
                rot_accel_z=float(row["rot_accel_z_rad_s2"]) if row.get("rot_accel_z_rad_s2") else 0.0,
            )
            steps.append(step)
    
    # Calculate average COM (typically constant, but average handles variations)
    if steps:
        avg_com = (
            sum(s.com_x_mm for s in steps) / len(steps),
            sum(s.com_y_mm for s in steps) / len(steps),
            sum(s.com_z_mm for s in steps) / len(steps),
        )
    else:
        avg_com = (0.0, 0.0, 0.0)
    
    return steps, avg_com


# =============================================================================
# PyQt5 GUI
# =============================================================================

def _set_modern_fusion_palette(app) -> None:
    """
    A simple modern dark-ish Fusion palette (matching ansys_inertia_relief_out_to_csv.py).
    """
    from PyQt5.QtGui import QPalette, QColor

    app.setStyle("Fusion")
    palette = QPalette()

    # Base
    palette.setColor(QPalette.Window, QColor(35, 35, 38))
    palette.setColor(QPalette.WindowText, QColor(220, 220, 220))
    palette.setColor(QPalette.Base, QColor(25, 25, 28))
    palette.setColor(QPalette.AlternateBase, QColor(35, 35, 38))
    palette.setColor(QPalette.ToolTipBase, QColor(220, 220, 220))
    palette.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
    palette.setColor(QPalette.Text, QColor(220, 220, 220))
    palette.setColor(QPalette.Button, QColor(45, 45, 48))
    palette.setColor(QPalette.ButtonText, QColor(220, 220, 220))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))

    # Highlights
    palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))

    # Disabled
    palette.setColor(QPalette.Disabled, QPalette.Text, QColor(120, 120, 120))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(120, 120, 120))

    app.setPalette(palette)


def generate_act_script(
    steps: List[IRLoadStep],
    avg_com: Tuple[float, float, float],
    use_mm_units: bool = True,
    cs_name: str = "CS_CenterOfMass",
    save_path: Optional[str] = None,
    close_after_save: bool = False,
) -> str:
    """
    Generate the ACT Python script text that can be pasted into Mechanical.
    
    Args:
        steps: List of IRLoadStep data
        avg_com: Average center of mass (x, y, z) in mm
        use_mm_units: If True, use mm/s² for translational acceleration
        cs_name: Name for the coordinate system
        save_path: Optional path to save .mechdat file
        close_after_save: If True, close Mechanical after saving
    
    Returns:
        Python script text for ACT console
    """
    # Header
    lines = [
        '"""',
        'Auto-generated ACT script: Apply Inertia Relief Accelerations',
        f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        f'Load Steps: {len(steps)}',
        f'COM: ({avg_com[0]:.4f}, {avg_com[1]:.4f}, {avg_com[2]:.4f}) mm',
        '"""',
        '',
        'import io',
        'import sys',
        '',
        '# =============================================================================',
        '# Step 1: Create Coordinate System at Center of Mass',
        '# =============================================================================',
        'model = ExtAPI.DataModel.Project.Model',
        'coord_systems = model.CoordinateSystems',
        '',
        'cs_com = coord_systems.AddCoordinateSystem()',
        f'cs_com.Name = "{cs_name}"',
        f'cs_com.OriginX = Quantity({avg_com[0]:.6g}, "mm")',
        f'cs_com.OriginY = Quantity({avg_com[1]:.6g}, "mm")',
        f'cs_com.OriginZ = Quantity({avg_com[2]:.6g}, "mm")',
        '',
        '# =============================================================================',
        '# Step 2: Get Static Structural Analysis',
        '# =============================================================================',
        'static_analysis = None',
        'for analysis in model.Analyses:',
        '    if "Static" in analysis.Name or analysis.AnalysisType == AnalysisType.Static:',
        '        static_analysis = analysis',
        '        break',
        '',
        'if static_analysis is None:',
        '    raise RuntimeError("No Static Structural analysis found!")',
        '',
        'print(f"Using analysis: {static_analysis.Name}")',
        '',
        '# =============================================================================',
        '# Step 3: Configure Load Steps',
        '# =============================================================================',
        'analysis_settings = static_analysis.AnalysisSettings',
        f'analysis_settings.NumberOfSteps = {len(steps)}',
        '',
    ]
    
    # Step end times
    for i, step in enumerate(steps):
        lines.append(f'analysis_settings.SetStepEndTime({i + 1}, Quantity({step.time_s:.6g}, "sec"))')
    
    lines.extend([
        '',
        '# =============================================================================',
        '# Step 4: Add Translational Acceleration BC',
        '# =============================================================================',
        'accel = static_analysis.AddAcceleration()',
        'accel.Name = "IR_Translational_Acceleration"',
        'accel.CoordinateSystem = cs_com',
        'accel.DefineBy = LoadDefineBy.Components',
        '',
        '# Tabular data: Time vs Acceleration',
        'times = [',
    ])
    
    # Times
    for step in steps:
        lines.append(f'    Quantity({step.time_s:.6g}, "sec"),')
    lines.append(']')
    
    # Acceleration values
    unit_factor = 1000.0 if use_mm_units else 1.0
    accel_unit = "mm/s^2" if use_mm_units else "m/s^2"
    
    lines.extend([
        '',
        'ax_values = [',
    ])
    for step in steps:
        lines.append(f'    Quantity({step.trans_accel_x * unit_factor:.6g}, "{accel_unit}"),')
    lines.append(']')
    
    lines.extend([
        '',
        'ay_values = [',
    ])
    for step in steps:
        lines.append(f'    Quantity({step.trans_accel_y * unit_factor:.6g}, "{accel_unit}"),')
    lines.append(']')
    
    lines.extend([
        '',
        'az_values = [',
    ])
    for step in steps:
        lines.append(f'    Quantity({step.trans_accel_z * unit_factor:.6g}, "{accel_unit}"),')
    lines.append(']')
    
    lines.extend([
        '',
        'accel.XComponent.Inputs[0].DiscreteValues = times',
        'accel.XComponent.Output.DiscreteValues = ax_values',
        'accel.YComponent.Inputs[0].DiscreteValues = times',
        'accel.YComponent.Output.DiscreteValues = ay_values',
        'accel.ZComponent.Inputs[0].DiscreteValues = times',
        'accel.ZComponent.Output.DiscreteValues = az_values',
        '',
        '# =============================================================================',
        '# Step 5: Add Rotational Acceleration BC',
        '# =============================================================================',
        'rot_accel = static_analysis.AddRotationalAcceleration()',
        'rot_accel.Name = "IR_Rotational_Acceleration"',
        'rot_accel.CoordinateSystem = cs_com',
        'rot_accel.DefineBy = LoadDefineBy.Components',
        '',
        '# Tabular data: Time vs Rotational Acceleration (rad/s^2)',
        'rx_values = [',
    ])
    for step in steps:
        lines.append(f'    Quantity({step.rot_accel_x:.6g}, "rad/s^2"),')
    lines.append(']')
    
    lines.extend([
        '',
        'ry_values = [',
    ])
    for step in steps:
        lines.append(f'    Quantity({step.rot_accel_y:.6g}, "rad/s^2"),')
    lines.append(']')
    
    lines.extend([
        '',
        'rz_values = [',
    ])
    for step in steps:
        lines.append(f'    Quantity({step.rot_accel_z:.6g}, "rad/s^2"),')
    lines.append(']')
    
    lines.extend([
        '',
        'rot_accel.XComponent.Inputs[0].DiscreteValues = times',
        'rot_accel.XComponent.Output.DiscreteValues = rx_values',
        'rot_accel.YComponent.Inputs[0].DiscreteValues = times',
        'rot_accel.YComponent.Output.DiscreteValues = ry_values',
        'rot_accel.ZComponent.Inputs[0].DiscreteValues = times',
        'rot_accel.ZComponent.Output.DiscreteValues = rz_values',
        '',
        '# =============================================================================',
        '# Step 6: Print Project Tree',
        '# =============================================================================',
        '',
        'def print_tree_act(obj=None, indent=0, max_depth=6):',
        '    """Print project tree structure (ACT console compatible)."""',
        '    if obj is None:',
        '        obj = ExtAPI.DataModel.Project',
        '    ',
        '    if indent > max_depth:',
        '        return',
        '    ',
        '    prefix = "|  " * indent + "|- " if indent > 0 else ""',
        '    name = getattr(obj, "Name", str(type(obj).__name__))',
        '    print(f"{prefix}{name}")',
        '    ',
        '    # Try to get children',
        '    children = []',
        '    if hasattr(obj, "Children"):',
        '        children = list(obj.Children)',
        '    elif hasattr(obj, "Model") and indent == 0:',
        '        children = [obj.Model]',
        '    ',
        '    for child in children:',
        '        print_tree_act(child, indent + 1, max_depth)',
        '',
        'print("")',
        'print("=" * 70)',
        'print("PROJECT TREE")',
        'print("=" * 70)',
        '',
        '# Capture tree output',
        '_tree_buffer = io.StringIO()',
        '_old_stdout = sys.stdout',
        'sys.stdout = _tree_buffer',
        'try:',
        '    print_tree_act()',
        'finally:',
        '    sys.stdout = _old_stdout',
        '',
        '_tree_output = _tree_buffer.getvalue()',
        'print(_tree_output)',
        '',
    ])
    
    # Save project if path provided
    if save_path:
        save_path_escaped = save_path.replace('\\', '\\\\')
        lines.extend([
            '# =============================================================================',
            '# Step 7: Save Project',
            '# =============================================================================',
            f'_save_path = r"{save_path_escaped}"',
            'print(f"Saving project to: {_save_path}")',
            '',
            '# Save using ExtAPI',
            'ExtAPI.DataModel.Project.Save(_save_path)',
            'print("Project saved successfully!")',
            '',
        ])
    
    # Success message
    lines.extend([
        '# =============================================================================',
        '# Success Summary',
        '# =============================================================================',
        'print("")',
        'print("=" * 70)',
        'print("INERTIA RELIEF BCs APPLIED SUCCESSFULLY!")',
        'print("=" * 70)',
        'print(f"  Coordinate System: {cs_com.Name}")',
        f'print(f"  Origin (COM): ({avg_com[0]:.4f}, {avg_com[1]:.4f}, {avg_com[2]:.4f}) mm")',
        'print(f"  Translational Accel BC: {accel.Name}")',
        'print(f"  Rotational Accel BC: {rot_accel.Name}")',
        f'print(f"  Load Steps: {len(steps)}")',
    ])
    
    if save_path:
        lines.append('print(f"  Saved to: {_save_path}")')
    
    lines.extend([
        'print("=" * 70)',
        '',
        '# Store tree output for potential GUI display',
        '_IR_TREE_OUTPUT = _tree_output',
    ])
    
    return '\n'.join(lines)


def generate_pymechanical_script(
    steps: List[IRLoadStep],
    avg_com: Tuple[float, float, float],
    use_mm_units: bool = True,
    cs_name: str = "CS_CenterOfMass",
    mechdat_path: Optional[str] = None,
    input_mechdat_path: Optional[str] = None,
    ansys_version: int = 232,
) -> str:
    """
    Generate a complete PyMechanical script that can be run standalone.
    
    This script launches Mechanical, opens an existing mechdat (if provided),
    applies the BCs, prints the tree, saves, and closes cleanly.
    """
    mechdat_path_escaped = mechdat_path.replace('\\', '\\\\') if mechdat_path else ""
    input_path_escaped = input_mechdat_path.replace('\\', '\\\\') if input_mechdat_path else ""
    
    lines = [
        '"""',
        'Auto-generated PyMechanical Script: Apply Inertia Relief Accelerations',
        f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        '',
        'This script launches Ansys Mechanical, applies IR boundary conditions,',
        'prints the project tree, saves, and closes cleanly.',
        '"""',
        '',
        'import io',
        'import sys',
        'from contextlib import redirect_stdout',
        '',
        '# =============================================================================',
        '# Launch PyMechanical',
        '# =============================================================================',
        'from ansys.mechanical.core import App',
        '',
        f'print("Launching Ansys Mechanical (version {ansys_version})...")',
        f'app = App(version={ansys_version})',
        'print(f"Mechanical launched: {{app}}")',
        '',
    ]
    
    # Open existing mechdat if provided
    if input_mechdat_path:
        lines.extend([
            '# Open existing project',
            f'input_path = r"{input_path_escaped}"',
            'print(f"Opening project: {input_path}")',
            'app.open(input_path)',
            '',
        ])
    
    lines.extend([
        '# Import global variables',
        'from ansys.mechanical.core.embedding.enum_importer import *  # noqa: F403',
        'from ansys.mechanical.core.embedding.global_importer import Quantity',
        '',
        '# Get ExtAPI reference',
        'ExtAPI = app.ExtAPI',
        'model = ExtAPI.DataModel.Project.Model',
        '',
        '# =============================================================================',
        '# Create Coordinate System at Center of Mass',
        '# =============================================================================',
        'coord_systems = model.CoordinateSystems',
        '',
        'cs_com = coord_systems.AddCoordinateSystem()',
        f'cs_com.Name = "{cs_name}"',
        f'cs_com.OriginX = Quantity({avg_com[0]:.6g}, "mm")',
        f'cs_com.OriginY = Quantity({avg_com[1]:.6g}, "mm")',
        f'cs_com.OriginZ = Quantity({avg_com[2]:.6g}, "mm")',
        'print(f"Created coordinate system: {cs_com.Name}")',
        '',
        '# =============================================================================',
        '# Get Static Structural Analysis',
        '# =============================================================================',
        'static_analysis = None',
        'for analysis in model.Analyses:',
        '    if "Static" in analysis.Name or analysis.AnalysisType == AnalysisType.Static:',
        '        static_analysis = analysis',
        '        break',
        '',
        'if static_analysis is None:',
        '    app.close()',
        '    raise RuntimeError("No Static Structural analysis found!")',
        '',
        'print(f"Using analysis: {static_analysis.Name}")',
        '',
        '# =============================================================================',
        '# Configure Load Steps',
        '# =============================================================================',
        'analysis_settings = static_analysis.AnalysisSettings',
        f'analysis_settings.NumberOfSteps = {len(steps)}',
        '',
    ])
    
    for i, step in enumerate(steps):
        lines.append(f'analysis_settings.SetStepEndTime({i + 1}, Quantity({step.time_s:.6g}, "sec"))')
    
    lines.extend([
        f'print(f"Configured {len(steps)} load steps")',
        '',
        '# =============================================================================',
        '# Add Translational Acceleration BC',
        '# =============================================================================',
        'accel = static_analysis.AddAcceleration()',
        'accel.Name = "IR_Translational_Acceleration"',
        'accel.CoordinateSystem = cs_com',
        'accel.DefineBy = LoadDefineBy.Components',
        '',
        'times = [',
    ])
    
    for step in steps:
        lines.append(f'    Quantity({step.time_s:.6g}, "sec"),')
    lines.append(']')
    
    unit_factor = 1000.0 if use_mm_units else 1.0
    accel_unit = "mm/s^2" if use_mm_units else "m/s^2"
    
    lines.extend(['', 'ax_values = ['])
    for step in steps:
        lines.append(f'    Quantity({step.trans_accel_x * unit_factor:.6g}, "{accel_unit}"),')
    lines.append(']')
    
    lines.extend(['', 'ay_values = ['])
    for step in steps:
        lines.append(f'    Quantity({step.trans_accel_y * unit_factor:.6g}, "{accel_unit}"),')
    lines.append(']')
    
    lines.extend(['', 'az_values = ['])
    for step in steps:
        lines.append(f'    Quantity({step.trans_accel_z * unit_factor:.6g}, "{accel_unit}"),')
    lines.append(']')
    
    lines.extend([
        '',
        'accel.XComponent.Inputs[0].DiscreteValues = times',
        'accel.XComponent.Output.DiscreteValues = ax_values',
        'accel.YComponent.Inputs[0].DiscreteValues = times',
        'accel.YComponent.Output.DiscreteValues = ay_values',
        'accel.ZComponent.Inputs[0].DiscreteValues = times',
        'accel.ZComponent.Output.DiscreteValues = az_values',
        'print(f"Created translational acceleration BC: {accel.Name}")',
        '',
        '# =============================================================================',
        '# Add Rotational Acceleration BC',
        '# =============================================================================',
        'rot_accel = static_analysis.AddRotationalAcceleration()',
        'rot_accel.Name = "IR_Rotational_Acceleration"',
        'rot_accel.CoordinateSystem = cs_com',
        'rot_accel.DefineBy = LoadDefineBy.Components',
        '',
        'rx_values = [',
    ])
    for step in steps:
        lines.append(f'    Quantity({step.rot_accel_x:.6g}, "rad/s^2"),')
    lines.append(']')
    
    lines.extend(['', 'ry_values = ['])
    for step in steps:
        lines.append(f'    Quantity({step.rot_accel_y:.6g}, "rad/s^2"),')
    lines.append(']')
    
    lines.extend(['', 'rz_values = ['])
    for step in steps:
        lines.append(f'    Quantity({step.rot_accel_z:.6g}, "rad/s^2"),')
    lines.append(']')
    
    lines.extend([
        '',
        'rot_accel.XComponent.Inputs[0].DiscreteValues = times',
        'rot_accel.XComponent.Output.DiscreteValues = rx_values',
        'rot_accel.YComponent.Inputs[0].DiscreteValues = times',
        'rot_accel.YComponent.Output.DiscreteValues = ry_values',
        'rot_accel.ZComponent.Inputs[0].DiscreteValues = times',
        'rot_accel.ZComponent.Output.DiscreteValues = rz_values',
        'print(f"Created rotational acceleration BC: {rot_accel.Name}")',
        '',
        '# =============================================================================',
        '# Print Project Tree (using app.print_tree)',
        '# =============================================================================',
        'print("")',
        'print("=" * 70)',
        'print("PROJECT TREE")',
        'print("=" * 70)',
        '',
        '# Capture tree output',
        'tree_buffer = io.StringIO()',
        'with redirect_stdout(tree_buffer):',
        '    app.print_tree()',
        '',
        'tree_output = tree_buffer.getvalue()',
        'print(tree_output)',
        '',
    ])
    
    # Save project
    if mechdat_path:
        lines.extend([
            '# =============================================================================',
            '# Save Project',
            '# =============================================================================',
            f'save_path = r"{mechdat_path_escaped}"',
            'print(f"Saving project to: {save_path}")',
            'app.save_as(save_path, overwrite=True)',
            'print("Project saved successfully!")',
            '',
        ])
    
    # Close Mechanical
    lines.extend([
        '# =============================================================================',
        '# Close Mechanical Session',
        '# =============================================================================',
        'print("")',
        'print("=" * 70)',
        'print("SUCCESS SUMMARY")',
        'print("=" * 70)',
        'print(f"  Coordinate System: {cs_com.Name}")',
        f'print(f"  Origin (COM): ({avg_com[0]:.4f}, {avg_com[1]:.4f}, {avg_com[2]:.4f}) mm")',
        'print(f"  Translational Accel BC: {accel.Name}")',
        'print(f"  Rotational Accel BC: {rot_accel.Name}")',
        f'print(f"  Load Steps: {len(steps)}")',
    ])
    
    if mechdat_path:
        lines.append('print(f"  Saved to: {save_path}")')
    
    lines.extend([
        'print("=" * 70)',
        '',
        '# Close the application',
        'print("")',
        'print("Closing Mechanical session...")',
        'app.close()',
        'print("Mechanical session closed successfully.")',
        '',
        '# Return tree output for potential GUI display',
        'TREE_OUTPUT = tree_output',
    ])
    
    return '\n'.join(lines)


class IRAccelerationApplicatorGUI:
    """PyQt5 GUI for applying inertia relief accelerations to Mechanical."""
    
    def __init__(self):
        from PyQt5.QtCore import Qt
        from PyQt5.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QDialog,
            QFileDialog,
            QGroupBox,
            QHBoxLayout,
            QHeaderView,
            QLabel,
            QLineEdit,
            QMessageBox,
            QPushButton,
            QSpinBox,
            QTableWidget,
            QTableWidgetItem,
            QTextEdit,
            QVBoxLayout,
            QSplitter,
            QTabWidget,
            QWidget,
        )
        
        self.app = QApplication.instance() or QApplication(sys.argv)
        _set_modern_fusion_palette(self.app)
        
        self.dlg = QDialog()
        self.dlg.setWindowTitle("Apply Inertia Relief Accelerations - PyMechanical Script Generator")
        self.dlg.setModal(True)
        self.dlg.resize(1200, 800)
        
        self.csv_path: Optional[Path] = None
        self.steps: List[IRLoadStep] = []
        self.avg_com: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        
        # === File Selection ===
        file_group = QGroupBox("1. Select IR Summary CSV")
        file_layout = QHBoxLayout()
        
        self.file_label = QLabel("No file selected.")
        self.file_label.setWordWrap(True)
        
        self.pick_btn = QPushButton("Browse...")
        self.pick_btn.clicked.connect(self._on_pick_file)
        
        file_layout.addWidget(self.file_label, 1)
        file_layout.addWidget(self.pick_btn)
        file_group.setLayout(file_layout)
        
        # === Options ===
        options_group = QGroupBox("2. Options")
        options_layout = QVBoxLayout()
        
        # Row 1: Unit system and CS name
        row1 = QHBoxLayout()
        self.use_mm_check = QCheckBox("Use mm/s^2 for translational acceleration (NMM unit system)")
        self.use_mm_check.setChecked(True)
        self.use_mm_check.stateChanged.connect(self._on_options_changed)
        
        cs_label = QLabel("CS Name:")
        self.cs_name_edit = QLineEdit("CS_CenterOfMass")
        self.cs_name_edit.setMaximumWidth(200)
        self.cs_name_edit.textChanged.connect(self._on_options_changed)
        
        row1.addWidget(self.use_mm_check)
        row1.addStretch(1)
        row1.addWidget(cs_label)
        row1.addWidget(self.cs_name_edit)
        
        # Row 2: Save options
        row2 = QHBoxLayout()
        self.save_mechdat_check = QCheckBox("Save .mechdat after applying:")
        self.save_mechdat_check.stateChanged.connect(self._on_save_option_changed)
        
        self.mechdat_path_edit = QLineEdit()
        self.mechdat_path_edit.setPlaceholderText("Path to save .mechdat file...")
        self.mechdat_path_edit.setEnabled(False)
        self.mechdat_path_edit.textChanged.connect(self._on_options_changed)
        
        self.mechdat_browse_btn = QPushButton("Browse...")
        self.mechdat_browse_btn.setEnabled(False)
        self.mechdat_browse_btn.clicked.connect(self._on_browse_mechdat)
        
        row2.addWidget(self.save_mechdat_check)
        row2.addWidget(self.mechdat_path_edit, 1)
        row2.addWidget(self.mechdat_browse_btn)
        
        # Row 3: Ansys version for PyMechanical
        row3 = QHBoxLayout()
        version_label = QLabel("Ansys Version (for PyMechanical):")
        self.version_spin = QSpinBox()
        self.version_spin.setRange(221, 260)
        self.version_spin.setValue(232)  # 2023 R2
        self.version_spin.setToolTip("232 = 2023 R2, 241 = 2024 R1, etc.")
        self.version_spin.valueChanged.connect(self._on_options_changed)
        
        row3.addWidget(version_label)
        row3.addWidget(self.version_spin)
        row3.addStretch(1)
        
        options_layout.addLayout(row1)
        options_layout.addLayout(row2)
        options_layout.addLayout(row3)
        options_group.setLayout(options_layout)
        
        # === Data Preview (Table) ===
        preview_layout = QVBoxLayout()
        
        self.com_label = QLabel("Center of Mass: -")
        preview_layout.addWidget(self.com_label)
        
        self.table = QTableWidget()
        self.table.setColumnCount(8)
        self.table.setHorizontalHeaderLabels([
            "Step", "Time (s)", 
            "Ax (m/s^2)", "Ay (m/s^2)", "Az (m/s^2)",
            "Rx (rad/s^2)", "Ry (rad/s^2)", "Rz (rad/s^2)"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        preview_layout.addWidget(self.table)
        
        preview_widget = QGroupBox("3. Data Preview")
        preview_widget.setLayout(preview_layout)
        
        # === Tabbed Script Views ===
        script_tabs = QTabWidget()
        
        # ACT Script Tab
        self.act_script_edit = QTextEdit()
        self.act_script_edit.setReadOnly(True)
        self.act_script_edit.setFontFamily("Consolas")
        self.act_script_edit.setPlaceholderText("Select a CSV file to generate the ACT script...")
        script_tabs.addTab(self.act_script_edit, "ACT Console Script")
        
        # PyMechanical Script Tab
        self.pymech_script_edit = QTextEdit()
        self.pymech_script_edit.setReadOnly(True)
        self.pymech_script_edit.setFontFamily("Consolas")
        self.pymech_script_edit.setPlaceholderText("Select a CSV file to generate the PyMechanical script...")
        script_tabs.addTab(self.pymech_script_edit, "PyMechanical Script")
        
        script_widget = QGroupBox("4. Generated Scripts")
        script_layout = QVBoxLayout()
        script_layout.addWidget(script_tabs)
        script_widget.setLayout(script_layout)
        
        # === Splitter for preview and script ===
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(preview_widget)
        splitter.addWidget(script_widget)
        splitter.setSizes([250, 400])
        
        # === Buttons ===
        btn_layout = QHBoxLayout()
        
        self.copy_act_btn = QPushButton("Copy ACT Script")
        self.copy_act_btn.clicked.connect(self._on_copy_act_script)
        self.copy_act_btn.setEnabled(False)
        
        self.copy_pymech_btn = QPushButton("Copy PyMechanical Script")
        self.copy_pymech_btn.clicked.connect(self._on_copy_pymech_script)
        self.copy_pymech_btn.setEnabled(False)
        
        self.save_btn = QPushButton("Save Script As...")
        self.save_btn.clicked.connect(self._on_save_script)
        self.save_btn.setEnabled(False)
        
        self.close_btn = QPushButton("Close")
        self.close_btn.clicked.connect(self.dlg.close)
        
        btn_layout.addWidget(self.copy_act_btn)
        btn_layout.addWidget(self.copy_pymech_btn)
        btn_layout.addWidget(self.save_btn)
        btn_layout.addStretch(1)
        btn_layout.addWidget(self.close_btn)
        
        # === Main Layout ===
        main_layout = QVBoxLayout()
        main_layout.addWidget(file_group)
        main_layout.addWidget(options_group)
        main_layout.addWidget(splitter, 1)
        main_layout.addLayout(btn_layout)
        
        self.dlg.setLayout(main_layout)
    
    def _on_pick_file(self) -> None:
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        
        path, _ = QFileDialog.getOpenFileName(
            self.dlg,
            "Select IR Summary CSV",
            "",
            "CSV files (*.csv);;All files (*.*)",
        )
        
        if not path:
            return
        
        self.csv_path = Path(path)
        self.file_label.setText(f"Selected: {self.csv_path.name}")
        
        # Set default mechdat path
        default_mechdat = self.csv_path.parent / (self.csv_path.stem + "_ir_applied.mechdat")
        self.mechdat_path_edit.setText(str(default_mechdat))
        
        try:
            self.steps, self.avg_com = read_ir_summary_csv(str(self.csv_path))
            self._update_preview()
            self._update_scripts()
            self.copy_act_btn.setEnabled(True)
            self.copy_pymech_btn.setEnabled(True)
            self.save_btn.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self.dlg, "Error", f"Failed to read CSV:\n{e}")
            self.steps = []
            self.avg_com = (0.0, 0.0, 0.0)
            self._update_preview()
    
    def _on_save_option_changed(self) -> None:
        enabled = self.save_mechdat_check.isChecked()
        self.mechdat_path_edit.setEnabled(enabled)
        self.mechdat_browse_btn.setEnabled(enabled)
        self._on_options_changed()
    
    def _on_browse_mechdat(self) -> None:
        from PyQt5.QtWidgets import QFileDialog
        
        start_dir = ""
        if self.csv_path:
            start_dir = str(self.csv_path.parent)
        
        path, _ = QFileDialog.getSaveFileName(
            self.dlg,
            "Save Mechanical Project As",
            start_dir,
            "Mechanical Database (*.mechdat);;All files (*.*)",
        )
        
        if path:
            self.mechdat_path_edit.setText(path)
    
    def _on_options_changed(self) -> None:
        if self.steps:
            self._update_scripts()
    
    def _update_preview(self) -> None:
        from PyQt5.QtWidgets import QTableWidgetItem
        
        self.com_label.setText(
            f"Center of Mass: ({self.avg_com[0]:.4f}, {self.avg_com[1]:.4f}, {self.avg_com[2]:.4f}) mm"
        )
        
        self.table.setRowCount(len(self.steps))
        
        for row, step in enumerate(self.steps):
            self.table.setItem(row, 0, QTableWidgetItem(str(step.load_step)))
            self.table.setItem(row, 1, QTableWidgetItem(f"{step.time_s:.2f}"))
            self.table.setItem(row, 2, QTableWidgetItem(f"{step.trans_accel_x:.4g}"))
            self.table.setItem(row, 3, QTableWidgetItem(f"{step.trans_accel_y:.4g}"))
            self.table.setItem(row, 4, QTableWidgetItem(f"{step.trans_accel_z:.4g}"))
            self.table.setItem(row, 5, QTableWidgetItem(f"{step.rot_accel_x:.4e}"))
            self.table.setItem(row, 6, QTableWidgetItem(f"{step.rot_accel_y:.4e}"))
            self.table.setItem(row, 7, QTableWidgetItem(f"{step.rot_accel_z:.4e}"))
    
    def _update_scripts(self) -> None:
        if not self.steps:
            self.act_script_edit.clear()
            self.pymech_script_edit.clear()
            return
        
        save_path = None
        if self.save_mechdat_check.isChecked() and self.mechdat_path_edit.text():
            save_path = self.mechdat_path_edit.text()
        
        # Generate ACT script
        act_script = generate_act_script(
            self.steps,
            self.avg_com,
            use_mm_units=self.use_mm_check.isChecked(),
            cs_name=self.cs_name_edit.text() or "CS_CenterOfMass",
            save_path=save_path,
        )
        self.act_script_edit.setPlainText(act_script)
        
        # Generate PyMechanical script
        pymech_script = generate_pymechanical_script(
            self.steps,
            self.avg_com,
            use_mm_units=self.use_mm_check.isChecked(),
            cs_name=self.cs_name_edit.text() or "CS_CenterOfMass",
            mechdat_path=save_path,
            ansys_version=self.version_spin.value(),
        )
        self.pymech_script_edit.setPlainText(pymech_script)
    
    def _on_copy_act_script(self) -> None:
        from PyQt5.QtWidgets import QApplication, QMessageBox
        
        script = self.act_script_edit.toPlainText()
        if script:
            QApplication.clipboard().setText(script)
            QMessageBox.information(
                self.dlg,
                "Copied!",
                "ACT script copied to clipboard.\n\n"
                "Paste into Mechanical's ACT console to apply.\n\n"
                "The script will:\n"
                "  - Create coordinate system at COM\n"
                "  - Add translational & rotational acceleration BCs\n"
                "  - Print the project tree\n"
                "  - Save the project (if path specified)"
            )
    
    def _on_copy_pymech_script(self) -> None:
        from PyQt5.QtWidgets import QApplication, QMessageBox
        
        script = self.pymech_script_edit.toPlainText()
        if script:
            QApplication.clipboard().setText(script)
            QMessageBox.information(
                self.dlg,
                "Copied!",
                "PyMechanical script copied to clipboard.\n\n"
                "Save to a .py file and run with Python to:\n"
                "  - Launch Ansys Mechanical\n"
                "  - Apply all boundary conditions\n"
                "  - Print the project tree (app.print_tree())\n"
                "  - Save the .mechdat file\n"
                "  - Close Mechanical cleanly"
            )
    
    def _on_save_script(self) -> None:
        from PyQt5.QtWidgets import QFileDialog, QMessageBox
        
        if not self.csv_path:
            return
        
        default_name = self.csv_path.stem + "_apply_ir.py"
        path, _ = QFileDialog.getSaveFileName(
            self.dlg,
            "Save Script",
            str(self.csv_path.parent / default_name),
            "Python files (*.py);;All files (*.*)",
        )
        
        if not path:
            return
        
        # Determine which script to save based on filename
        script = self.pymech_script_edit.toPlainText()
        script_type = "PyMechanical"
        
        if "_act" in path.lower():
            script = self.act_script_edit.toPlainText()
            script_type = "ACT"
        
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(script)
            QMessageBox.information(
                self.dlg,
                "Saved!",
                f"{script_type} script saved to:\n{path}\n\n"
                f"{'Run with Python' if script_type == 'PyMechanical' else 'Paste into Mechanical ACT console'} to apply."
            )
        except Exception as e:
            QMessageBox.critical(self.dlg, "Error", f"Failed to save:\n{e}")
    
    def run(self) -> int:
        self.dlg.show()
        return self.app.exec_()


def run_gui() -> int:
    """Launch the PyQt5 GUI."""
    gui = IRAccelerationApplicatorGUI()
    return gui.run()


# =============================================================================
# ACT Scripting Functions (for direct use inside Mechanical)
# =============================================================================

def create_coordinate_system_at_com(com_mm: Tuple[float, float, float], cs_name: str = "CS_COM") -> "CoordinateSystem":
    """Create a local Cartesian coordinate system at the Center of Mass."""
    model = ExtAPI.DataModel.Project.Model  # noqa: F821
    coord_systems = model.CoordinateSystems
    
    cs = coord_systems.AddCoordinateSystem()
    cs.Name = cs_name
    cs.OriginX = Quantity(com_mm[0], "mm")  # noqa: F821
    cs.OriginY = Quantity(com_mm[1], "mm")  # noqa: F821
    cs.OriginZ = Quantity(com_mm[2], "mm")  # noqa: F821
    
    return cs


def setup_static_analysis_steps(analysis, num_steps: int, step_times: List[float]) -> None:
    """Configure the analysis to have the correct number of steps."""
    analysis_settings = analysis.AnalysisSettings
    analysis_settings.NumberOfSteps = num_steps
    
    for i, time_s in enumerate(step_times):
        step_num = i + 1
        analysis_settings.SetStepEndTime(step_num, Quantity(time_s, "sec"))  # noqa: F821


def add_acceleration_bc(
    analysis,
    steps: List[IRLoadStep],
    cs: "CoordinateSystem",
    use_mm_units: bool = True
) -> "Acceleration":
    """Add an Acceleration boundary condition with tabular data."""
    accel = analysis.AddAcceleration()
    accel.Name = "IR_Translational_Acceleration"
    accel.CoordinateSystem = cs
    accel.DefineBy = LoadDefineBy.Components  # noqa: F821
    
    unit_factor = 1000.0 if use_mm_units else 1.0
    accel_unit = "mm/s^2" if use_mm_units else "m/s^2"
    
    times = [Quantity(s.time_s, "sec") for s in steps]  # noqa: F821
    ax_values = [Quantity(s.trans_accel_x * unit_factor, accel_unit) for s in steps]  # noqa: F821
    ay_values = [Quantity(s.trans_accel_y * unit_factor, accel_unit) for s in steps]  # noqa: F821
    az_values = [Quantity(s.trans_accel_z * unit_factor, accel_unit) for s in steps]  # noqa: F821
    
    accel.XComponent.Inputs[0].DiscreteValues = times
    accel.XComponent.Output.DiscreteValues = ax_values
    accel.YComponent.Inputs[0].DiscreteValues = times
    accel.YComponent.Output.DiscreteValues = ay_values
    accel.ZComponent.Inputs[0].DiscreteValues = times
    accel.ZComponent.Output.DiscreteValues = az_values
    
    return accel


def add_rotational_acceleration_bc(
    analysis,
    steps: List[IRLoadStep],
    cs: "CoordinateSystem",
) -> "RotationalAcceleration":
    """Add a Rotational Acceleration boundary condition with tabular data."""
    rot_accel = analysis.AddRotationalAcceleration()
    rot_accel.Name = "IR_Rotational_Acceleration"
    rot_accel.CoordinateSystem = cs
    rot_accel.DefineBy = LoadDefineBy.Components  # noqa: F821
    
    rot_unit = "rad/s^2"
    
    times = [Quantity(s.time_s, "sec") for s in steps]  # noqa: F821
    rx_values = [Quantity(s.rot_accel_x, rot_unit) for s in steps]  # noqa: F821
    ry_values = [Quantity(s.rot_accel_y, rot_unit) for s in steps]  # noqa: F821
    rz_values = [Quantity(s.rot_accel_z, rot_unit) for s in steps]  # noqa: F821
    
    rot_accel.XComponent.Inputs[0].DiscreteValues = times
    rot_accel.XComponent.Output.DiscreteValues = rx_values
    rot_accel.YComponent.Inputs[0].DiscreteValues = times
    rot_accel.YComponent.Output.DiscreteValues = ry_values
    rot_accel.ZComponent.Inputs[0].DiscreteValues = times
    rot_accel.ZComponent.Output.DiscreteValues = rz_values
    
    return rot_accel


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    # Check if we're running inside Mechanical
    try:
        ExtAPI  # noqa: F821 - ExtAPI is defined in Mechanical environment
        print("Running inside Mechanical. Use the generated script directly.")
    except NameError:
        # Running standalone - launch GUI
        sys.exit(run_gui())
