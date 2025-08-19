# mos_app.py
# PyQt5 + NumPy Margin-of-Safety calculator with material-temperature interpolation
# Default MoS% = [1 - sigma_max / YTS(T)] * 100  (FS optional via allowable = YTS/FS)

import sys, os, json, re
import numpy as np
import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets

APP_NAME = "MoS Calculator"
VERSION = "0.1.0"

# ------------------------- utilities -------------------------
def c_to_k(c): return c + 273.15
def k_to_c(k): return k - 273.15

def interp1_linear(x, xp, fp, extrap='clamp'):
    """Vectorized 1D linear interpolation. xp must be increasing."""
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    x = np.asarray(x, dtype=float)

    if xp.ndim != 1 or fp.ndim != 1 or xp.size != fp.size:
        raise ValueError("xp and fp must be 1D and same length")
    if np.any(np.diff(xp) <= 0):
        raise ValueError("xp must be strictly increasing")

    idx = np.searchsorted(xp, x, side='left')
    idx = np.clip(idx, 1, len(xp)-1)
    x0 = xp[idx-1]; x1 = xp[idx]
    y0 = fp[idx-1]; y1 = fp[idx]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (x - x0) / (x1 - x0)
    y = y0 + t*(y1 - y0)

    if extrap == 'clamp':
        y = np.where(x <= xp[0], fp[0], np.where(x >= xp[-1], fp[-1], y))
    elif extrap == 'error':
        mask = (x < xp[0]) | (x > xp[-1])
        if np.any(mask):
            raise ValueError("Extrapolation outside material temperature range")
    elif extrap == 'linear':
        pass
    else:
        raise ValueError("Unknown extrapolation mode")
    return y

def guess_col(df, patterns):
    pats = [re.compile(p, re.I) for p in patterns]
    for col in df.columns:
        name = str(col)
        for p in pats:
            if p.search(name):
                return col
    return None

# ------------------------- pandas model -------------------------
class PandasModel(QtCore.QAbstractTableModel):
    def __init__(self, df=pd.DataFrame(), parent=None):
        super().__init__(parent)
        self._df = df

    def setDataFrame(self, df):
        self.beginResetModel()
        self._df = df.copy()
        self.endResetModel()

    def rowCount(self, parent=None): return 0 if parent and parent.isValid() else len(self._df.index)
    def columnCount(self, parent=None): return 0 if parent and parent.isValid() else len(self._df.columns)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid(): return None
        if role in (QtCore.Qt.DisplayRole, QtCore.Qt.EditRole):
            val = self._df.iat[index.row(), index.column()]
            if isinstance(val, float):
                return f"{val:.6g}"
            return str(val)
        return None

    def headerData(self, section, orientation, role):
        if role != QtCore.Qt.DisplayRole: return None
        if orientation == QtCore.Qt.Horizontal: return str(self._df.columns[section])
        return str(self._df.index[section])

# ------------------------- workers -------------------------
class ComputeWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(pd.DataFrame, dict, str)  # df, stats, err
    def __init__(self, material, ansys, map_mat, map_ans, units, fs, extrap, percent=True):
        super().__init__()
        self.material = material  # DataFrame
        self.ansys = ansys        # DataFrame
        self.map_mat = map_mat    # dict: temp, yts
        self.map_ans = map_ans    # dict: node, stress, temp
        self.units = units        # dict: mat_T in {"K","C"}, ans_T in {"K","C"}
        self.fs = fs
        self.extrap = extrap
        self.percent = percent

    @QtCore.pyqtSlot()
    def run(self):
        try:
            # extract arrays
            Tm = self.material[self.map_mat['temp']].to_numpy(dtype=float)
            YTS = self.material[self.map_mat['yts']].to_numpy(dtype=float)

            # unit normalize to Celsius internally
            if self.units['mat_T'] == 'K': Tm = k_to_c(Tm)

            # sort material by T
            order = np.argsort(Tm)
            Tm = Tm[order]; YTS = YTS[order]

            # ansys arrays
            node = self.ansys[self.map_ans['node']].to_numpy()
            stress = self.ansys[self.map_ans['stress']].to_numpy(dtype=float)
            Ta = self.ansys[self.map_ans['temp']].to_numpy(dtype=float)
            if self.units['ans_T'] == 'K': Ta = k_to_c(Ta)

            # interpolate
            yts_at_T = interp1_linear(Ta, Tm, YTS, extrap=self.extrap)
            allowable = yts_at_T / max(self.fs, 1e-12)

            # MoS
            with np.errstate(divide='ignore', invalid='ignore'):
                mos_ratio = 1.0 - (stress / allowable)  # e.g. 0.2 means 20% margin
            mos = 100.0 * mos_ratio if self.percent else mos_ratio

            df_out = pd.DataFrame({
                "NodeID": node,
                "Stress_MPa": stress,
                "Temp_C": Ta,
                "YTS_at_T_MPa": yts_at_T,
                "Allowable_MPa": allowable,
                "MoS_%": mos if self.percent else None,
                "MoS": None if self.percent else mos_ratio
            })
            if self.percent:
                df_out["MoS_%"] = mos
            else:
                df_out.drop(columns=["MoS_%"], inplace=True)

            # stats
            bad = np.sum((mos if self.percent else mos_ratio*100.0) < 0.0)
            worst_idx = np.argmin(mos if self.percent else mos_ratio*100.0)
            stats = {
                "rows": int(df_out.shape[0]),
                "num_negative": int(bad),
                "worst_node": str(df_out.iloc[worst_idx]["NodeID"]),
                "worst_mos_percent": float(mos[worst_idx] if self.percent else mos_ratio[worst_idx]*100.0)
            }
            self.finished.emit(df_out, stats, "")
        except Exception as e:
            self.finished.emit(pd.DataFrame(), {}, str(e))

# ------------------------- main window -------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{VERSION}")
        self.resize(1200, 760)

        self.material_df = pd.DataFrame()
        self.ansys_df = pd.DataFrame()
        self.results_df = pd.DataFrame()

        self._build_ui()
        self._apply_style()

    # UI
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # toolbar
        tb = QtWidgets.QToolBar()
        tb.setMovable(False)
        self.addToolBar(tb)
        act_dark = QtWidgets.QAction("Toggle Theme", self)
        act_dark.triggered.connect(self._apply_style)
        tb.addAction(act_dark)

        # stacked panels
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setTabPosition(QtWidgets.QTabWidget.West)
        layout.addWidget(self.tabs)

        self._build_tab_material()
        self._build_tab_ansys()
        self._build_tab_compute()
        self._build_tab_results()

        # status
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

    def _build_tab_material(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        # controls
        ctrl = QtWidgets.QHBoxLayout()
        self.btn_mat = QtWidgets.QPushButton("Load Material CSV/XLSX")
        self.btn_mat.clicked.connect(self.load_material)
        ctrl.addWidget(self.btn_mat)

        ctrl.addStretch()
        ctrl.addWidget(QtWidgets.QLabel("Material T units:"))
        self.combo_mat_units = QtWidgets.QComboBox()
        self.combo_mat_units.addItems(["C", "K"])
        self.combo_mat_units.setCurrentText("C")
        ctrl.addWidget(self.combo_mat_units)

        v.addLayout(ctrl)

        # column mapping
        form = QtWidgets.QFormLayout()
        self.cb_mat_temp = QtWidgets.QComboBox()
        self.cb_mat_yts = QtWidgets.QComboBox()
        form.addRow("Temperature column", self.cb_mat_temp)
        form.addRow("Yield (YTS) column", self.cb_mat_yts)
        v.addLayout(form)

        # preview
        self.tbl_mat = QtWidgets.QTableView()
        self.model_mat = PandasModel()
        self.tbl_mat.setModel(self.model_mat)
        self.tbl_mat.setSortingEnabled(True)
        v.addWidget(self.tbl_mat)

        self.tabs.addTab(w, "Material DB")

    def _build_tab_ansys(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        ctrl = QtWidgets.QHBoxLayout()
        self.btn_ans = QtWidgets.QPushButton("Load ANSYS CSV/XLSX")
        self.btn_ans.clicked.connect(self.load_ansys)
        ctrl.addWidget(self.btn_ans)

        ctrl.addStretch()
        ctrl.addWidget(QtWidgets.QLabel("ANSYS T units:"))
        self.combo_ans_units = QtWidgets.QComboBox()
        self.combo_ans_units.addItems(["C", "K"])
        self.combo_ans_units.setCurrentText("C")
        ctrl.addWidget(self.combo_ans_units)
        v.addLayout(ctrl)

        form = QtWidgets.QFormLayout()
        self.cb_ans_node = QtWidgets.QComboBox()
        self.cb_ans_stress = QtWidgets.QComboBox()
        self.cb_ans_temp = QtWidgets.QComboBox()
        form.addRow("Node ID column", self.cb_ans_node)
        form.addRow("Von Mises Stress column", self.cb_ans_stress)
        form.addRow("Temperature column", self.cb_ans_temp)
        v.addLayout(form)

        self.tbl_ans = QtWidgets.QTableView()
        self.model_ans = PandasModel()
        self.tbl_ans.setModel(self.model_ans)
        self.tbl_ans.setSortingEnabled(True)
        v.addWidget(self.tbl_ans)

        self.tabs.addTab(w, "ANSYS Import")

    def _build_tab_compute(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        form = QtWidgets.QFormLayout()
        self.fs_spin = QtWidgets.QDoubleSpinBox()
        self.fs_spin.setRange(0.01, 1000.0)
        self.fs_spin.setDecimals(3)
        self.fs_spin.setValue(1.0)
        form.addRow("Factor of Safety (applied to YTS):", self.fs_spin)

        self.combo_extrap = QtWidgets.QComboBox()
        self.combo_extrap.addItems(["clamp", "linear", "error"])
        form.addRow("Extrapolation policy:", self.combo_extrap)

        self.chk_percent = QtWidgets.QCheckBox("Output MoS in percent")
        self.chk_percent.setChecked(True)
        form.addRow("", self.chk_percent)

        v.addLayout(form)

        self.btn_compute = QtWidgets.QPushButton("Compute MoS")
        self.btn_compute.clicked.connect(self.compute)
        v.addWidget(self.btn_compute)

        self.lbl_summary = QtWidgets.QLabel("Summary: N/A")
        v.addWidget(self.lbl_summary)

        self.tabs.addTab(w, "Compute")

    def _build_tab_results(self):
        w = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(w)

        top = QtWidgets.QHBoxLayout()
        self.btn_export = QtWidgets.QPushButton("Export CSV")
        self.btn_export.clicked.connect(self.export_csv)
        top.addWidget(self.btn_export)

        self.chk_only_neg = QtWidgets.QCheckBox("Show MoS < 0 only")
        self.chk_only_neg.toggled.connect(self._apply_filter)
        top.addWidget(self.chk_only_neg)

        top.addStretch()
        v.addLayout(top)

        self.tbl_out = QtWidgets.QTableView()
        self.model_out = PandasModel()
        self.proxy = QtCore.QSortFilterProxyModel(self)
        self.proxy.setSourceModel(self.model_out)
        self.tbl_out.setModel(self.proxy)
        self.tbl_out.setSortingEnabled(True)
        v.addWidget(self.tbl_out)

        self.tabs.addTab(w, "Results")

    def _apply_style(self):
        # simple light/dark toggle based on current palette
        pal = self.palette()
        is_dark = pal.color(QtGui.QPalette.Window).value() > 128
        if is_dark:
            self.setStyleSheet("""
                QMainWindow, QWidget { background: #101012; color: #e5e5e5; }
                QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTableView, QTextEdit {
                    background: #1a1a1f; color: #e5e5e5; border: 1px solid #2a2a2f; }
                QPushButton { background: #2a2a35; border: 1px solid #3a3a45; padding: 6px 10px; border-radius: 8px; }
                QPushButton:hover { background: #343444; }
                QHeaderView::section { background: #22222a; padding: 4px; border: none; }
                QTabWidget::pane { border: 1px solid #2a2a2f; }
                QTabBar::tab { padding: 8px; }
            """)
        else:
            self.setStyleSheet("""
                QTableView { gridline-color: #ddd; }
                QPushButton { padding: 6px 10px; border-radius: 8px; }
            """)

    # IO
    def _read_any(self, path):
        ext = os.path.splitext(path)[1].lower()
        if ext in (".csv", ".txt"):
            return pd.read_csv(path)
        elif ext in (".xlsx", ".xls"):
            return pd.read_excel(path)
        else:
            raise ValueError("Use CSV or XLSX")

    def load_material(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Material File", "", "Data (*.csv *.xlsx)")
        if not path: return
        try:
            df = self._read_any(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e)); return
        self.material_df = df
        self.model_mat.setDataFrame(df)
        self.cb_mat_temp.clear(); self.cb_mat_yts.clear()
        cols = [str(c) for c in df.columns]
        self.cb_mat_temp.addItems(cols); self.cb_mat_yts.addItems(cols)

        # auto guess
        t_guess = guess_col(df, [r"temp", r"T\(C|K\)"])
        y_guess = guess_col(df, [r"yield", r"YTS", r"proof", r"Rp0[.,]2"])
        if t_guess: self.cb_mat_temp.setCurrentText(str(t_guess))
        if y_guess: self.cb_mat_yts.setCurrentText(str(y_guess))
        self.status.showMessage(f"Loaded material file: {os.path.basename(path)}", 5000)

    def load_ansys(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open ANSYS File", "", "Data (*.csv *.xlsx)")
        if not path: return
        try:
            df = self._read_any(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e)); return
        self.ansys_df = df
        self.model_ans.setDataFrame(df)
        self.cb_ans_node.clear(); self.cb_ans_temp.clear(); self.cb_ans_stress.clear()
        cols = [str(c) for c in df.columns]
        self.cb_ans_node.addItems(cols); self.cb_ans_stress.addItems(cols); self.cb_ans_temp.addItems(cols)

        # auto guess
        node_g = guess_col(df, [r"node", r"nid", r"id"])
        stress_g = guess_col(df, [r"vm", r"von", r"mises", r"stress"])
        temp_g = guess_col(df, [r"temp", r"T\(C|K\)"])
        if node_g: self.cb_ans_node.setCurrentText(str(node_g))
        if stress_g: self.cb_ans_stress.setCurrentText(str(stress_g))
        if temp_g: self.cb_ans_temp.setCurrentText(str(temp_g))
        self.status.showMessage(f"Loaded ANSYS file: {os.path.basename(path)}", 5000)

    # compute
    def compute(self):
        if self.material_df.empty or self.ansys_df.empty:
            QtWidgets.QMessageBox.warning(self, "Missing data", "Load both material and ANSYS files first.")
            return

        map_mat = {"temp": self.cb_mat_temp.currentText(), "yts": self.cb_mat_yts.currentText()}
        map_ans = {
            "node": self.cb_ans_node.currentText(),
            "stress": self.cb_ans_stress.currentText(),
            "temp": self.cb_ans_temp.currentText()
        }
        units = {"mat_T": self.combo_mat_units.currentText(), "ans_T": self.combo_ans_units.currentText()}
        fs = float(self.fs_spin.value())
        extrap = self.combo_extrap.currentText()
        percent = self.chk_percent.isChecked()

        # thread
        self.btn_compute.setEnabled(False)
        self.status.showMessage("Computing…")
        self.thread = QtCore.QThread()
        self.worker = ComputeWorker(self.material_df, self.ansys_df, map_mat, map_ans, units, fs, extrap, percent)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_compute_done)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()

    @QtCore.pyqtSlot(pd.DataFrame, dict, str)
    def _on_compute_done(self, df, stats, err):
        self.btn_compute.setEnabled(True)
        if err:
            QtWidgets.QMessageBox.critical(self, "Compute error", err)
            self.status.showMessage("Error", 5000)
            return
        self.results_df = df
        self.model_out.setDataFrame(df)
        self._apply_filter()
        s = f"Rows: {stats['rows']}, MoS<0: {stats['num_negative']}, Worst Node: {stats['worst_node']}, Worst MoS%: {stats['worst_mos_percent']:.2f}"
        self.lbl_summary.setText("Summary: " + s)
        self.status.showMessage("Done", 3000)
        self.tabs.setCurrentIndex(3)

    def _apply_filter(self):
        if self.results_df.empty:
            self.proxy.setFilterRegularExpression(QtCore.QRegularExpression())
            return
        colname = "MoS_%" if "MoS_%" in self.results_df.columns else "MoS"
        col = list(self.results_df.columns).index(colname)
        self.proxy.setFilterKeyColumn(col)
        if self.chk_only_neg.isChecked():
            # custom negative filter
            self.proxy.setFilterRegularExpression(QtCore.QRegularExpression(r"^-"))
        else:
            self.proxy.setFilterRegularExpression(QtCore.QRegularExpression())

    def export_csv(self):
        if self.results_df.empty:
            QtWidgets.QMessageBox.information(self, "No data", "Nothing to export.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Results CSV", "mos_results.csv", "CSV (*.csv)")
        if not path: return
        try:
            self.results_df.to_csv(path, index=False)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e)); return
        self.status.showMessage(f"Saved: {os.path.basename(path)}", 5000)

# ------------------------- main -------------------------
def main():
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
