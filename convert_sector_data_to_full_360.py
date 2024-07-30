import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QComboBox, QPushButton, QSpinBox, QWidget, QProgressBar
import numpy as np
import os

class DataProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Data Processing App")
        self.setGeometry(100, 100, 400, 300)

        self.layout = QVBoxLayout()

        self.file_label = QLabel("Select a file:", self)
        self.layout.addWidget(self.file_label)

        self.file_button = QPushButton("Browse", self)
        self.file_button.clicked.connect(self.open_file_dialog)
        self.layout.addWidget(self.file_button)

        self.separator_label = QLabel("Select separator:", self)
        self.layout.addWidget(self.separator_label)

        self.separator_combobox = QComboBox(self)
        self.separator_combobox.addItems([",", ";", "\\t", " "])
        self.layout.addWidget(self.separator_combobox)

        self.start_line_label = QLabel("Start reading from line:", self)
        self.layout.addWidget(self.start_line_label)

        self.start_line_spinbox = QSpinBox(self)
        self.start_line_spinbox.setMinimum(1)
        self.layout.addWidget(self.start_line_spinbox)

        self.axis_label = QLabel("Select axis for circular pattern:", self)
        self.layout.addWidget(self.axis_label)

        self.axis_combobox = QComboBox(self)
        self.axis_combobox.addItems(["X", "Y", "Z"])
        self.layout.addWidget(self.axis_combobox)

        self.sectors_label = QLabel("Number of sectors (including original):", self)
        self.layout.addWidget(self.sectors_label)

        self.sectors_input = QSpinBox(self)
        self.sectors_input.setMinimum(2)  # Minimum 2 sectors to include the original and one pattern
        self.layout.addWidget(self.sectors_input)

        self.process_button = QPushButton("Process Data", self)
        self.process_button.clicked.connect(self.process_data)
        self.layout.addWidget(self.process_button)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        self.layout.addWidget(self.progress_bar)

        container = QWidget()
        container.setLayout(self.layout)
        self.setCentralWidget(container)

    def open_file_dialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Data File", "", "All Files (*);;CSV Files (*.csv);;Text Files (*.txt);;DAT Files (*.dat)", options=options)
        if file_name:
            self.file_label.setText(f"Selected File: {file_name}")
            self.file_path = file_name

    def process_data(self):
        try:
            separator = self.separator_combobox.currentText()
            start_line = self.start_line_spinbox.value() - 1
            axis = self.axis_combobox.currentText()
            sectors = self.sectors_input.value()

            if separator == "\\t":
                separator = "\t"

            if hasattr(self, 'file_path'):
                df = pd.read_csv(self.file_path, sep=separator, skiprows=start_line, header=None, dtype=np.float32)
                df.columns = ['X', 'Y', 'Z', 'Field Data']
            else:
                df = self.create_sample_data()

            repeated_data = self.circular_patterning(df, axis, sectors)
            self.save_data(repeated_data)
        except Exception as e:
            self.file_label.setText(f"Error: {str(e)}")

    def circular_patterning(self, df, axis, sectors):
        angle_step = 360 / sectors
        angles = np.arange(0, 360, angle_step, dtype=np.float32)
        n_angles = len(angles)

        x, y, z = df['X'].values, df['Y'].values, df['Z'].values

        repeated_data_list = []

        for angle in angles:
            sin_angle = np.sin(np.deg2rad(angle))
            cos_angle = np.cos(np.deg2rad(angle))
            
            if axis == "X":
                new_y = y * cos_angle - z * sin_angle
                new_z = y * sin_angle + z * cos_angle
                new_x = x
            elif axis == "Y":
                new_x = x * cos_angle + z * sin_angle
                new_z = -x * sin_angle + z * cos_angle
                new_y = y
            else:  # Z axis
                new_x = x * cos_angle - y * sin_angle
                new_y = x * sin_angle + y * cos_angle
                new_z = z

            repeated_data = pd.DataFrame({
                'X': new_x,
                'Y': new_y,
                'Z': new_z,
                'Field Data': df['Field Data'].values
            })

            repeated_data_list.append(repeated_data)

        all_repeated_data = pd.concat(repeated_data_list, ignore_index=True)

        self.progress_bar.setValue(100)
        return all_repeated_data

    def save_data(self, data):
        base_name = os.path.splitext(self.file_path)[0] if hasattr(self, 'file_path') else 'sample_data'
        output_file = f"{base_name}_full_360_data.csv"
        data.to_csv(output_file, index=False)
        self.file_label.setText(f"Data saved to: {output_file}")
        self.progress_bar.setValue(100)

    def create_sample_data(self):
        data = {
            'X': np.array([1, 2, 3], dtype=np.float32),
            'Y': np.array([4, 5, 6], dtype=np.float32),
            'Z': np.array([7, 8, 9], dtype=np.float32),
            'Field Data': np.array([10, 20, 30], dtype=np.float32)
        }
        return pd.DataFrame(data)

def main():
    app = QApplication(sys.argv)
    window = DataProcessingApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
