import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QComboBox, QPushButton, QSpinBox, QWidget
import numpy as np
import os

class DataProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Data Processing App")
        self.setGeometry(100, 100, 400, 200)

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

        self.process_button = QPushButton("Process Data", self)
        self.process_button.clicked.connect(self.process_data)
        self.layout.addWidget(self.process_button)

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

            if separator == "\\t":
                separator = "\t"

            df = pd.read_csv(self.file_path, sep=separator, skiprows=start_line, header=None)
            df.columns = ['X', 'Y', 'Z', 'Field Data']

            repeated_data = self.circular_patterning(df, axis)
            self.save_data(repeated_data)
        except Exception as e:
            self.file_label.setText(f"Error: {str(e)}")

    def circular_patterning(self, df, axis):
        angle_step = 360 / len(df)
        angles = np.arange(0, 360, angle_step)

        if axis == "X":
            repeated_data = pd.concat([df.assign(Y=df['Y'] * np.cos(np.deg2rad(angle)) - df['Z'] * np.sin(np.deg2rad(angle)),
                                                 Z=df['Y'] * np.sin(np.deg2rad(angle)) + df['Z'] * np.cos(np.deg2rad(angle)))
                                       for angle in angles])
        elif axis == "Y":
            repeated_data = pd.concat([df.assign(X=df['X'] * np.cos(np.deg2rad(angle)) + df['Z'] * np.sin(np.deg2rad(angle)),
                                                 Z=-df['X'] * np.sin(np.deg2rad(angle)) + df['Z'] * np.cos(np.deg2rad(angle)))
                                       for angle in angles])
        else:
            repeated_data = pd.concat([df.assign(X=df['X'] * np.cos(np.deg2rad(angle)) - df['Y'] * np.sin(np.deg2rad(angle)),
                                                 Y=df['X'] * np.sin(np.deg2rad(angle)) + df['Y'] * np.cos(np.deg2rad(angle)))
                                       for angle in angles])

        return repeated_data

    def save_data(self, data):
        base_name = os.path.splitext(self.file_path)[0]
        output_file = f"{base_name}_full_360_data.csv"
        data.to_csv(output_file, index=False)
        self.file_label.setText(f"Data saved to: {output_file}")

def main():
    app = QApplication(sys.argv)
    window = DataProcessingApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
