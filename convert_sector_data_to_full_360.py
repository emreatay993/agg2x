import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QComboBox, QPushButton, QSpinBox, QWidget, QHBoxLayout

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
        self.separator_combobox.addItems([",", ";", "\t", " "])
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

            df = pd.read_csv(self.file_path, sep=separator, skiprows=start_line, header=None)
            df.columns = ['X', 'Y', 'Z', 'Field Data']

            self.circular_patterning(df, axis)
        except Exception as e:
            self.file_label.setText(f"Error: {str(e)}")

    def circular_patterning(self, df, axis):
        import numpy as np

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

        print(repeated_data)
        # Here you can save or further process the repeated_data as needed

def main():
    app = QApplication(sys.argv)
    window = DataProcessingApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()