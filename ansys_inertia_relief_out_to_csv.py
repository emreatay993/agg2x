"""
ANSYS MAPDL solve.out inertia-relief extractor (multi-load-step) -> CSV.

What it does
------------
For each load step in a MAPDL solve output file, if the load step is solved with
Inertia Relief (Static), it extracts:
  - Center of Mass coordinates (X,Y,Z) -> output in mm
  - Translational and rotational accelerations (about COM) as printed
  - (Also) forces at COM and moments about COM as printed

The script is unit-aware. It reads the unit system from the standard "MPA UNITS"
block (LENGTH/MASS/TIME/FORCE). It then converts outputs to:
  - COM: mm
  - Forces: N
  - Moments: N·mm

UI
--
If no input paths are provided on the command line, a PyQt5 file dialog is shown
to select one or more .out files. A CSV is written next to each input file.

If the detailed output CSV is locked/open (e.g. in Excel) and cannot be
overwritten, the program halts.
"""

from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


FLOAT_RE = re.compile(r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][+-]?\d+)?")


def _floats(s: str) -> List[float]:
    return [float(x) for x in FLOAT_RE.findall(s)]


@dataclass(frozen=True)
class UnitSystem:
    length: Optional[str] = None  # e.g. "mm", "m"
    mass: Optional[str] = None    # e.g. "Mg", "kg"
    time: Optional[str] = None    # e.g. "sec"
    force: Optional[str] = None   # e.g. "N"

    def length_to_mm(self) -> float:
        if not self.length:
            return 1.0
        u = self.length.strip()
        # Case-sensitive first (ANSYS uses "Mg" for tonne)
        u_l = u.lower()
        mapping = {
            "mm": 1.0,
            "millimeter": 1.0,
            "millimeters": 1.0,
            "cm": 10.0,
            "centimeter": 10.0,
            "centimeters": 10.0,
            "m": 1000.0,
            "meter": 1000.0,
            "meters": 1000.0,
            "in": 25.4,
            "inch": 25.4,
            "inches": 25.4,
            "ft": 304.8,
            "foot": 304.8,
            "feet": 304.8,
        }
        return mapping.get(u_l, 1.0)

    def length_to_m(self) -> float:
        return self.length_to_mm() / 1000.0

    def time_to_s(self) -> float:
        if not self.time:
            return 1.0
        u = self.time.strip()
        u_l = u.lower()
        mapping = {
            "s": 1.0,
            "sec": 1.0,
            "secs": 1.0,
            "second": 1.0,
            "seconds": 1.0,
            "ms": 1e-3,
            "msec": 1e-3,
            "msecs": 1e-3,
            "millisecond": 1e-3,
            "milliseconds": 1e-3,
            "min": 60.0,
            "mins": 60.0,
            "minute": 60.0,
            "minutes": 60.0,
            "hr": 3600.0,
            "hrs": 3600.0,
            "hour": 3600.0,
            "hours": 3600.0,
        }
        return mapping.get(u_l, 1.0)

    def force_to_n(self) -> float:
        if not self.force:
            return 1.0
        u = self.force.strip()
        u_l = u.lower()
        mapping = {
            "n": 1.0,
            "kn": 1000.0,
            "mn": 1_000_000.0,
            "lbf": 4.4482216152605,
        }
        return mapping.get(u_l, 1.0)

    def mass_to_kg(self) -> float:
        if not self.mass:
            return 1.0
        u = self.mass.strip()
        # Case sensitive first to avoid Mg vs mg ambiguity
        mapping_case = {
            "Mg": 1000.0,  # megagram = tonne
        }
        if u in mapping_case:
            return mapping_case[u]
        u_l = u.lower()
        mapping = {
            "kg": 1.0,
            "g": 1e-3,
            "mg": 1e-6,  # milligram
            "tonne": 1000.0,
            "t": 1000.0,
            "lbm": 0.45359237,
            "slug": 14.59390294,
        }
        return mapping.get(u_l, 1.0)


def parse_units(lines: Sequence[str]) -> UnitSystem:
    """
    Parse the standard units block in solve.out.

    Example:
      LENGTH      = MILLIMETERS (mm)
      MASS        = TONNE (Mg)
      TIME        = SECONDS (sec)
      FORCE       = NEWTON (N)
    """
    length = mass = time = force = None

    # Helpful fallback from the "consistent NMM" banner
    for line in lines[:500]:
        if "consistent" in line.lower() and "nmm" in line.lower():
            length = length or "mm"
            force = force or "N"
            # mass in N-mm unit systems is typically Mg (tonne)
            mass = mass or "Mg"
            time = time or "sec"
            break

    unit_line_re = re.compile(r"^\s*(LENGTH|MASS|TIME|FORCE)\s*=\s*.*\(([^)]+)\)\s*$", re.IGNORECASE)
    for line in lines:
        m = unit_line_re.match(line)
        if not m:
            continue
        key = m.group(1).upper()
        unit = m.group(2).strip()
        if key == "LENGTH" and length is None:
            length = unit
        elif key == "MASS" and mass is None:
            mass = unit
        elif key == "TIME" and time is None:
            time = unit
        elif key == "FORCE" and force is None:
            force = unit
        if length and mass and time and force:
            break

    return UnitSystem(length=length, mass=mass, time=time, force=force)


@dataclass
class LoadStepResult:
    load_step: int
    inertia_relief: bool
    time_end_raw: Optional[float] = None  # in-file time units
    time_end_s: Optional[float] = None    # seconds

    # Parsed (raw, in-file units)
    com_raw: Optional[Tuple[float, float, float]] = None
    total_mass_raw: Optional[float] = None
    inertia_tensor_com_raw: Optional[List[List[float]]] = None  # 3x3

    forces_at_com_raw: Optional[Tuple[float, float, float]] = None
    moments_about_com_raw: Optional[Tuple[float, float, float]] = None

    trans_accel_raw: Optional[Tuple[float, float, float]] = None
    rot_accel_raw: Optional[Tuple[float, float, float]] = None

    # Converted outputs
    com_out: Optional[Tuple[float, float, float]] = None
    forces_at_com_n: Optional[Tuple[float, float, float]] = None
    moments_about_com_out: Optional[Tuple[float, float, float]] = None

    trans_accel_m_s2: Optional[Tuple[float, float, float]] = None
    rot_accel_rad_s2: Optional[Tuple[float, float, float]] = None


def _find_load_step_markers(lines: Sequence[str]) -> List[Tuple[int, int]]:
    markers: List[Tuple[int, int]] = []
    for i, line in enumerate(lines):
        if "LOAD STEP NUMBER" not in line:
            continue
        m = re.search(r"(\d+)\s*$", line)
        if not m:
            continue
        markers.append((int(m.group(1)), i))
    return markers


def _parse_inertia_relief_flag(step_lines: Sequence[str]) -> Optional[bool]:
    for line in step_lines[:250]:
        if "INERTIA RELIEF" not in line:
            continue
        # Avoid matching global "INERTIA RELIEF KEY= ..."
        if "KEY" in line.upper():
            continue
        m = re.search(r"\bINERTIA RELIEF\.\s*.*\b(ON|OFF)\b", line, re.IGNORECASE)
        if not m:
            continue
        return m.group(1).upper() == "ON"
    return None


def _parse_last_vec3(step_lines: Sequence[str], contains: str) -> Optional[Tuple[float, float, float]]:
    last: Optional[Tuple[float, float, float]] = None
    contains_u = contains.upper()
    for line in step_lines:
        if contains_u not in line.upper():
            continue
        vals = _floats(line)
        if len(vals) >= 3:
            last = (vals[-3], vals[-2], vals[-1])
    return last


def _parse_last_com(step_lines: Sequence[str]) -> Optional[Tuple[float, float, float]]:
    last: Optional[Tuple[float, float, float]] = None
    for line in step_lines:
        # Avoid accidental matches like:
        #   "ROTATIONAL ACCELERATIONS ABOUT CENTER OF MASS ..."
        s = line.strip().upper()
        if not s.startswith("CENTER OF MASS"):
            continue
        vals = _floats(line)
        if len(vals) >= 3:
            last = (vals[0], vals[1], vals[2])
    return last


def _parse_last_total_mass(step_lines: Sequence[str]) -> Optional[float]:
    last: Optional[float] = None
    for line in step_lines:
        s = line.strip()
        if not s.upper().startswith("TOTAL MASS"):
            continue
        if "MATRIX" in s.upper():
            continue
        vals = _floats(line)
        if vals:
            last = vals[0]
    return last


def _parse_time_end(step_lines: Sequence[str]) -> Optional[float]:
    """
    Parse:
      TIME AT END OF THE LOAD STEP. . . . . . . . . .  1.0000
    """
    for line in step_lines[:250]:
        if "TIME AT END OF THE LOAD STEP" not in line.upper():
            continue
        vals = _floats(line)
        if vals:
            return vals[-1]
    return None


def _parse_last_inertia_tensor_about_com(step_lines: Sequence[str]) -> Optional[List[List[float]]]:
    headers = (
        "TOTAL INERTIA ABOUT CENTER OF MASS",
        "INERTIA TENSOR (I) ABOUT CENTER OF MASS",
        "MOMENTS AND PRODUCTS OF INERTIA TENSOR (I) ABOUT CENTER OF MASS",
    )

    last_mat: Optional[List[List[float]]] = None
    for i, line in enumerate(step_lines):
        u = line.upper()
        if not any(h in u for h in headers):
            continue

        floats: List[float] = []
        # collect from the following lines until we have 9 floats
        for j in range(i + 1, min(i + 25, len(step_lines))):
            floats.extend(_floats(step_lines[j]))
            if len(floats) >= 9:
                break
        if len(floats) >= 9:
            last_mat = [
                [floats[0], floats[1], floats[2]],
                [floats[3], floats[4], floats[5]],
                [floats[6], floats[7], floats[8]],
            ]
    return last_mat


def _mat_vec3(mat: List[List[float]], v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    x, y, z = v
    return (
        mat[0][0] * x + mat[0][1] * y + mat[0][2] * z,
        mat[1][0] * x + mat[1][1] * y + mat[1][2] * z,
        mat[2][0] * x + mat[2][1] * y + mat[2][2] * z,
    )


def parse_solve_out(text: str) -> Tuple[UnitSystem, List[LoadStepResult]]:
    lines = text.splitlines()
    units = parse_units(lines)

    markers = _find_load_step_markers(lines)
    if not markers:
        return units, []

    results: List[LoadStepResult] = []
    for idx, (ls, start) in enumerate(markers):
        end = markers[idx + 1][1] if idx + 1 < len(markers) else len(lines)
        step_lines = lines[start:end]

        ir = _parse_inertia_relief_flag(step_lines)
        inertia_relief = bool(ir)  # default False if missing

        r = LoadStepResult(load_step=ls, inertia_relief=inertia_relief)
        r.time_end_raw = _parse_time_end(step_lines)

        # Raw extractions
        r.com_raw = _parse_last_com(step_lines)
        r.total_mass_raw = _parse_last_total_mass(step_lines)
        r.inertia_tensor_com_raw = _parse_last_inertia_tensor_about_com(step_lines)

        r.forces_at_com_raw = _parse_last_vec3(step_lines, "FORCES AT CENTER OF MASS")
        r.moments_about_com_raw = _parse_last_vec3(step_lines, "MOMENTS ABOUT CENTER OF MASS")
        r.trans_accel_raw = _parse_last_vec3(step_lines, "TRANSLATIONAL ACCELERATIONS")
        r.rot_accel_raw = _parse_last_vec3(step_lines, "ROTATIONAL ACCELERATIONS ABOUT CENTER OF MASS")

        # Unit conversions
        L_to_m = units.length_to_m()
        L_to_mm = units.length_to_mm()
        T_to_s = units.time_to_s()
        F_to_N = units.force_to_n()

        if r.time_end_raw is not None:
            r.time_end_s = r.time_end_raw * T_to_s

        if r.forces_at_com_raw is not None:
            r.forces_at_com_n = (
                r.forces_at_com_raw[0] * F_to_N,
                r.forces_at_com_raw[1] * F_to_N,
                r.forces_at_com_raw[2] * F_to_N,
            )

        if r.trans_accel_raw is not None:
            # length / time^2 -> m / s^2
            scale = L_to_m / (T_to_s * T_to_s)
            r.trans_accel_m_s2 = (
                r.trans_accel_raw[0] * scale,
                r.trans_accel_raw[1] * scale,
                r.trans_accel_raw[2] * scale,
            )

        if r.rot_accel_raw is not None:
            # 1 / time^2 -> rad / s^2 (rad is dimensionless)
            scale = 1.0 / (T_to_s * T_to_s)
            r.rot_accel_rad_s2 = (
                r.rot_accel_raw[0] * scale,
                r.rot_accel_raw[1] * scale,
                r.rot_accel_raw[2] * scale,
            )

        results.append(r)

    return units, results


def _fmt(x: Optional[float]) -> str:
    if x is None:
        return ""
    # preserve exponential notation for big/small values
    return f"{x:.12g}"


@dataclass(frozen=True)
class OutputUnitSystem:
    """
    Output system selection for reporting length and moment units.
    Forces are always reported in Newtons.
    """
    name: str  # "Nmm" or "Nm"
    length_unit: str  # "mm" or "m"
    moment_unit: str  # "Nmm" or "Nm"


OUTPUT_UNITS_NMM = OutputUnitSystem(name="Nmm", length_unit="mm", moment_unit="Nmm")
OUTPUT_UNITS_NM = OutputUnitSystem(name="Nm", length_unit="m", moment_unit="Nm")


def apply_output_units(units: UnitSystem, out_units: OutputUnitSystem, r: LoadStepResult) -> None:
    """
    Populate r.com_out and r.moments_about_com_out based on the output unit selection.
    """
    if r.com_raw is not None:
        if out_units.name == "Nm":
            L = units.length_to_m()
        else:
            L = units.length_to_mm()
        r.com_out = (r.com_raw[0] * L, r.com_raw[1] * L, r.com_raw[2] * L)

    if r.moments_about_com_raw is not None:
        F_to_N = units.force_to_n()
        if out_units.name == "Nm":
            L = units.length_to_m()
        else:
            L = units.length_to_mm()
        r.moments_about_com_out = (
            r.moments_about_com_raw[0] * F_to_N * L,
            r.moments_about_com_raw[1] * F_to_N * L,
            r.moments_about_com_raw[2] * F_to_N * L,
        )


def write_csv(out_path: Path, in_path: Path, units: UnitSystem, out_units: OutputUnitSystem, results: Sequence[LoadStepResult]) -> None:
    cols = [
        "source_file",
        "load_step",
        "inertia_relief",
        "time_end_s",
        "units_length",
        "units_mass",
        "units_time",
        "units_force",
        f"com_x_{out_units.length_unit}",
        f"com_y_{out_units.length_unit}",
        f"com_z_{out_units.length_unit}",
        "forces_com_x_N",
        "forces_com_y_N",
        "forces_com_z_N",
        f"moments_com_x_{out_units.moment_unit}",
        f"moments_com_y_{out_units.moment_unit}",
        f"moments_com_z_{out_units.moment_unit}",
        "trans_accel_x_m_s2",
        "trans_accel_y_m_s2",
        "trans_accel_z_m_s2",
        "rot_accel_x_rad_s2",
        "rot_accel_y_rad_s2",
        "rot_accel_z_rad_s2",
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            if not r.inertia_relief:
                continue
            apply_output_units(units, out_units, r)
            row: Dict[str, str] = {
                "source_file": str(in_path),
                "load_step": str(r.load_step),
                "inertia_relief": "ON" if r.inertia_relief else "OFF",
                "time_end_s": _fmt(r.time_end_s),
                "units_length": units.length or "",
                "units_mass": units.mass or "",
                "units_time": units.time or "",
                "units_force": units.force or "",
            }

            if r.com_out:
                row.update(
                    **{
                        f"com_x_{out_units.length_unit}": _fmt(r.com_out[0]),
                        f"com_y_{out_units.length_unit}": _fmt(r.com_out[1]),
                        f"com_z_{out_units.length_unit}": _fmt(r.com_out[2]),
                    }
                )
            if r.forces_at_com_n:
                row.update(
                    forces_com_x_N=_fmt(r.forces_at_com_n[0]),
                    forces_com_y_N=_fmt(r.forces_at_com_n[1]),
                    forces_com_z_N=_fmt(r.forces_at_com_n[2]),
                )
            if r.moments_about_com_out:
                row.update(
                    **{
                        f"moments_com_x_{out_units.moment_unit}": _fmt(r.moments_about_com_out[0]),
                        f"moments_com_y_{out_units.moment_unit}": _fmt(r.moments_about_com_out[1]),
                        f"moments_com_z_{out_units.moment_unit}": _fmt(r.moments_about_com_out[2]),
                    }
                )
            if r.trans_accel_m_s2:
                row.update(
                    trans_accel_x_m_s2=_fmt(r.trans_accel_m_s2[0]),
                    trans_accel_y_m_s2=_fmt(r.trans_accel_m_s2[1]),
                    trans_accel_z_m_s2=_fmt(r.trans_accel_m_s2[2]),
                )
            if r.rot_accel_rad_s2:
                row.update(
                    rot_accel_x_rad_s2=_fmt(r.rot_accel_rad_s2[0]),
                    rot_accel_y_rad_s2=_fmt(r.rot_accel_rad_s2[1]),
                    rot_accel_z_rad_s2=_fmt(r.rot_accel_rad_s2[2]),
                )

            w.writerow(row)


def _set_modern_fusion_palette(app) -> None:
    """
    A simple modern dark-ish Fusion palette (similar to other tools in this repo).
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


def select_out_files_and_units_via_dialog() -> Tuple[List[Path], OutputUnitSystem]:
    """
    Small Fusion-style dialog:
      - choose output unit system (SI N·mm or SI N·m)
      - pick one or more .out files via QFileDialog
    """
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import (
        QApplication,
        QComboBox,
        QDialog,
        QFileDialog,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QVBoxLayout,
    )

    app = QApplication.instance() or QApplication(sys.argv)
    _set_modern_fusion_palette(app)

    dlg = QDialog()
    dlg.setWindowTitle("ANSYS Inertia Relief Extractor")
    dlg.setModal(True)

    chosen_paths: List[Path] = []

    units_label = QLabel("Output unit system:")
    units_combo = QComboBox()
    units_combo.addItem("SI (N·mm)", OUTPUT_UNITS_NMM)
    units_combo.addItem("SI (N·m)", OUTPUT_UNITS_NM)
    units_combo.setCurrentIndex(0)

    files_label = QLabel("No files selected.")
    files_label.setWordWrap(True)

    def on_pick_files() -> None:
        nonlocal chosen_paths
        paths, _ = QFileDialog.getOpenFileNames(
            dlg,
            "Select ANSYS solve output files (.out)",
            "",
            "ANSYS output (*.out);;All files (*.*)",
        )
        chosen_paths = [Path(p) for p in paths if p]
        if not chosen_paths:
            files_label.setText("No files selected.")
        elif len(chosen_paths) == 1:
            files_label.setText(f"Selected: {chosen_paths[0].name}")
        else:
            files_label.setText(f"Selected: {len(chosen_paths)} files")

    pick_btn = QPushButton("Select .out file(s)...")
    pick_btn.clicked.connect(on_pick_files)  # type: ignore[attr-defined]

    ok_btn = QPushButton("Run")
    ok_btn.setDefault(True)
    cancel_btn = QPushButton("Cancel")

    def on_ok() -> None:
        if chosen_paths:
            dlg.accept()

    ok_btn.clicked.connect(on_ok)  # type: ignore[attr-defined]
    cancel_btn.clicked.connect(dlg.reject)  # type: ignore[attr-defined]

    top = QHBoxLayout()
    top.addWidget(units_label)
    top.addWidget(units_combo, 1)

    btns = QHBoxLayout()
    btns.addStretch(1)
    btns.addWidget(cancel_btn)
    btns.addWidget(ok_btn)

    layout = QVBoxLayout()
    layout.addLayout(top)
    layout.addWidget(pick_btn, alignment=Qt.AlignLeft)
    layout.addWidget(files_label)
    layout.addStretch(1)
    layout.addLayout(btns)
    dlg.setLayout(layout)
    dlg.resize(720, 190)

    if dlg.exec_() != QDialog.Accepted:
        return [], OUTPUT_UNITS_NMM

    out_units: OutputUnitSystem = units_combo.currentData()
    return chosen_paths, out_units


def write_ir_input_summary_csv(out_path: Path, units: UnitSystem, results: Sequence[LoadStepResult]) -> None:
    """
    Create the simplified CSV requested for upstream IR inputs:
      load_step, time_end_s, com(mm), trans_accel(m/s^2), rot_accel(rad/s^2)
    """
    cols = [
        "load_step",
        "time_end_s",
        "com_x_mm",
        "com_y_mm",
        "com_z_mm",
        "trans_accel_x_m_s2",
        "trans_accel_y_m_s2",
        "trans_accel_z_m_s2",
        "rot_accel_x_rad_s2",
        "rot_accel_y_rad_s2",
        "rot_accel_z_rad_s2",
    ]

    L_to_mm = units.length_to_mm()

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            if not r.inertia_relief:
                continue
            com_mm = None
            if r.com_raw is not None:
                com_mm = (r.com_raw[0] * L_to_mm, r.com_raw[1] * L_to_mm, r.com_raw[2] * L_to_mm)
            row: Dict[str, str] = {
                "load_step": str(r.load_step),
                "time_end_s": _fmt(r.time_end_s),
                "com_x_mm": _fmt(com_mm[0]) if com_mm else "",
                "com_y_mm": _fmt(com_mm[1]) if com_mm else "",
                "com_z_mm": _fmt(com_mm[2]) if com_mm else "",
            }
            if r.trans_accel_m_s2:
                row.update(
                    trans_accel_x_m_s2=_fmt(r.trans_accel_m_s2[0]),
                    trans_accel_y_m_s2=_fmt(r.trans_accel_m_s2[1]),
                    trans_accel_z_m_s2=_fmt(r.trans_accel_m_s2[2]),
                )
            if r.rot_accel_rad_s2:
                row.update(
                    rot_accel_x_rad_s2=_fmt(r.rot_accel_rad_s2[0]),
                    rot_accel_y_rad_s2=_fmt(r.rot_accel_rad_s2[1]),
                    rot_accel_z_rad_s2=_fmt(r.rot_accel_rad_s2[2]),
                )
            w.writerow(row)


def process_file(path: Path, out_units: OutputUnitSystem) -> Tuple[Path, Path, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    units, results = parse_solve_out(text)
    out_path = path.with_suffix("").with_name(path.stem + "_inertia_relief.csv")
    # If the file is locked (PermissionError), let it raise -> caller halts.
    write_csv(out_path, path, units, out_units, results)

    # No datetime in filename; keep it unique per input file.
    summary_path = path.with_name(f"ansys_IR_input_summary_{path.stem}.csv")
    write_ir_input_summary_csv(summary_path, units, results)
    count_ir = sum(1 for r in results if r.inertia_relief)
    return out_path, summary_path, count_ir


def main(argv: Sequence[str]) -> int:
    args = [a for a in argv[1:] if not a.startswith("-")]
    paths: List[Path]
    out_units = OUTPUT_UNITS_NMM
    if args:
        paths = [Path(a) for a in args]
    else:
        paths, out_units = select_out_files_and_units_via_dialog()

    if not paths:
        return 0

    wrote: List[Tuple[Path, Path, int]] = []
    for p in paths:
        if not p.exists():
            print(f"[skip] not found: {p}")
            continue
        try:
            out_csv, summary_csv, n = process_file(p, out_units)
            wrote.append((out_csv, summary_csv, n))
            print(
                f"[ok] {p.name} -> {out_csv.name} and {summary_csv.name} "
                f"({n} inertia-relief load steps)"
            )
        except PermissionError as e:
            msg = (
                f"Permission denied while writing output CSV(s).\n\n"
                f"Close the CSV file(s) (e.g., Excel) and run again.\n\n"
                f"Details: {e}"
            )
            print(f"[error] {p}: {msg}")
            if not args:
                try:
                    from PyQt5.QtWidgets import QApplication, QMessageBox

                    app = QApplication.instance()
                    if app is not None:
                        QMessageBox.critical(None, "Cannot write CSV", msg)
                except Exception:
                    pass
            return 1
        except Exception as e:
            print(f"[error] {p}: {e}")

    # Nice completion message if we ran via dialog (i.e., no CLI args)
    if not args:
        try:
            from PyQt5.QtWidgets import QApplication, QMessageBox

            app = QApplication.instance()
            if app is not None:
                if wrote:
                    msg = "\n".join(
                        [f"{outp.name} + {summaryp.name}  (IR load steps: {n})" for outp, summaryp, n in wrote]
                    )
                    QMessageBox.information(None, "Done", f"CSV exported:\n{msg}")
                else:
                    QMessageBox.warning(None, "Nothing written", "No valid .out files were processed.")
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))


