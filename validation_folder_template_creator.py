"""
Validation Folder Template Creator

A PyQt6 GUI tool for creating a structured validation project folder template.

Install dependency:
    pip install PyQt6

Run:
    python validation_folder_template_creator.py
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


PROJECT_FOLDERS = [
    "00_Project_Admin_and_Index",
    "01_Requirements_and_Basis/Specifications",
    "01_Requirements_and_Basis/Acceptance_Criteria",
    "01_Requirements_and_Basis/Load_Cases",
    "01_Requirements_and_Basis/Interface_Definitions",
    "01_Requirements_and_Basis/Standards_and_References",
    "02_Global_FEA_Models/Baseline_Model",
    "02_Global_FEA_Models/Common_Material_Data",
    "02_Global_FEA_Models/Common_Boundary_Conditions",
    "02_Global_FEA_Models/Common_Load_Definitions",
    "02_Global_FEA_Models/Released_Model_Versions",
    "03_Tests",
    "04_Cross_Test_Correlation/Global_Correlation_Matrix",
    "04_Cross_Test_Correlation/Common_Sensor_Comparison",
    "04_Cross_Test_Correlation/Repeated_Load_Case_Comparison",
    "04_Cross_Test_Correlation/Model_Update_Tracking",
    "04_Cross_Test_Correlation/Lessons_Learned",
    "05_Final_Validation_Deliverables/Draft",
    "05_Final_Validation_Deliverables/Review",
    "05_Final_Validation_Deliverables/Released",
    "05_Final_Validation_Deliverables/Signed",
    "06_Scripts_and_Tools/Python",
    "06_Scripts_and_Tools/APDL",
    "06_Scripts_and_Tools/Matlab",
    "06_Scripts_and_Tools/Excel_Macros",
    "06_Scripts_and_Tools/Common_Postprocessing_Tools",
    "99_Archive/Superseded",
    "99_Archive/Obsolete",
    "99_Archive/Backup",
]


TEST_SUBFOLDERS = [
    "00_Test_Index",
    "01_Test_Planning/Test_Plans",
    "01_Test_Planning/Test_Procedures",
    "01_Test_Planning/Instrumentation_Plans",
    "01_Test_Planning/Sensor_Layouts",
    "01_Test_Planning/Risk_Assessments",
    "02_Raw_Test_Data/Original_From_Test_Rig",
    "02_Raw_Test_Data/Subcontractor_Data",
    "02_Raw_Test_Data/Strain_Gauge_Data",
    "02_Raw_Test_Data/Displacement_Data",
    "02_Raw_Test_Data/Load_Cell_Data",
    "02_Raw_Test_Data/Temperature_Data",
    "03_Processed_Test_Data/Cleaned_Data",
    "03_Processed_Test_Data/Filtered_Data",
    "03_Processed_Test_Data/Calculated_Channels",
    "03_Processed_Test_Data/Plots",
    "03_Processed_Test_Data/Data_Reduction_Scripts",
    "04_Test_Specific_FEA/Models",
    "04_Test_Specific_FEA/Mesh_Files",
    "04_Test_Specific_FEA/Material_Data",
    "04_Test_Specific_FEA/Boundary_Conditions",
    "04_Test_Specific_FEA/Load_Application",
    "04_Test_Specific_FEA/Solver_Input_Files",
    "05_FEA_Results/Raw_Results",
    "05_FEA_Results/Postprocessed_Results",
    "05_FEA_Results/Stress_Results",
    "05_FEA_Results/Strain_Results",
    "05_FEA_Results/Displacement_Results",
    "05_FEA_Results/Contact_Results",
    "05_FEA_Results/Result_Images",
    "06_Test_vs_FEA_Correlation/Correlation_Tables",
    "06_Test_vs_FEA_Correlation/SG_vs_FEA",
    "06_Test_vs_FEA_Correlation/Displacement_vs_FEA",
    "06_Test_vs_FEA_Correlation/Load_Response_Comparison",
    "06_Test_vs_FEA_Correlation/Error_Calculations",
    "06_Test_vs_FEA_Correlation/Correlation_Plots",
    "07_Photos_Videos/Test_Setup",
    "07_Photos_Videos/During_Test",
    "07_Photos_Videos/After_Test",
    "08_Subcontractor_Communication/Emails",
    "08_Subcontractor_Communication/Meeting_Minutes",
    "08_Subcontractor_Communication/Data_Deliveries",
    "08_Subcontractor_Communication/Questions_and_Answers",
    "08_Subcontractor_Communication/Subcontractor_Reports",
    "09_Presentations/Working_Presentations",
    "09_Presentations/Review_Board_Presentations",
    "09_Presentations/Customer_Presentations",
    "09_Presentations/Final_Presentations",
    "10_Test_Report/Draft",
    "10_Test_Report/Review",
    "10_Test_Report/Released",
    "10_Test_Report/Signed",
]


PROJECT_REGISTER_TEMPLATES = {
    "Project_Index.csv": [
        "File Name",
        "Folder Path",
        "Description",
        "Source",
        "Date Received",
        "Author",
        "Version",
        "Status",
        "Related Test",
        "Related Load Case",
        "Related FEA Model",
        "Used in Report?",
        "Notes",
    ],
    "Test_Matrix.csv": [
        "Test ID",
        "Test Name",
        "Component",
        "Purpose",
        "Load Case",
        "Requirement Reference",
        "Acceptance Criteria",
        "Test Date",
        "Test Facility",
        "Subcontractor",
        "FEA Model Version",
        "Report Status",
        "Result Status",
        "Comments",
    ],
    "Document_Register.csv": [
        "Document ID",
        "Document Name",
        "Folder Path",
        "Owner",
        "Version",
        "Status",
        "Release Date",
        "Notes",
    ],
    "Action_Item_List.csv": [
        "Action ID",
        "Description",
        "Owner",
        "Due Date",
        "Status",
        "Closure Evidence",
        "Notes",
    ],
    "Validation_Summary_Status.csv": [
        "Test ID",
        "Test Name",
        "Requirement",
        "Result",
        "Report Status",
        "Open Issues",
        "Notes",
    ],
}


TEST_REGISTER_TEMPLATES = {
    "{test_id}_Index.csv": [
        "File Name",
        "Description",
        "Folder",
        "Source",
        "Author",
        "Date",
        "Version",
        "Status",
        "Used in Report?",
        "Related Load Case",
        "Related Sensor",
        "Related FEA Model",
        "Notes",
    ],
    "{test_id}_Correlation_Index.csv": [
        "Sensor ID",
        "Measurement Type",
        "Physical Location",
        "Test Channel Name",
        "FEA Node/Element/Path",
        "Load Case",
        "Measured Value",
        "FEA Value",
        "Difference %",
        "Acceptance Limit",
        "Pass/Fail",
        "Comment",
    ],
}


README_TEXT = """# Validation Project Folder

This folder was created using the Validation Folder Template Creator.

## Basic Usage Rules

1. Do not modify raw test data.
2. Keep processed data traceable to raw data.
3. Keep FEA results traceable to the model version, load case, and boundary condition set.
4. Keep reports and presentations traceable to the data and analysis version used.
5. Avoid vague file names such as `final`, `latest`, `new`, or `updated`.

## Recommended File Naming

`YYYYMMDD_Project_Component_TestType_Content_vXX_Status.ext`

Example:

`20260605_CompressorCasing_StaticTest_FilteredSGData_v02_Working.xlsx`
"""


@dataclass
class CreationOptions:
    destination: Path
    project_name: str
    tests: list[str]
    create_registers: bool
    create_readme: bool


def sanitize_folder_name(name: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    cleaned = "".join("_" if c in invalid_chars else c for c in name.strip())
    cleaned = "_".join(cleaned.split())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def write_csv_template(path: Path, headers: Iterable[str]) -> None:
    if path.exists():
        return

    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(list(headers))


class FolderCreationWorker(QThread):
    progress_changed = pyqtSignal(int, int, str)
    finished_successfully = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, options: CreationOptions) -> None:
        super().__init__()
        self.options = options

    def run(self) -> None:
        try:
            project_folder = self.options.destination / sanitize_folder_name(self.options.project_name)

            folder_paths: list[Path] = []

            for folder in PROJECT_FOLDERS:
                folder_paths.append(project_folder / folder)

            for test in self.options.tests:
                test_folder_name = sanitize_folder_name(test)
                for subfolder in TEST_SUBFOLDERS:
                    folder_paths.append(project_folder / "03_Tests" / test_folder_name / subfolder)

            total_steps = len(folder_paths)

            if self.options.create_registers:
                total_steps += len(PROJECT_REGISTER_TEMPLATES)
                total_steps += len(self.options.tests) * len(TEST_REGISTER_TEMPLATES)

            if self.options.create_readme:
                total_steps += 1

            completed = 0

            project_folder.mkdir(parents=True, exist_ok=True)

            for folder_path in folder_paths:
                folder_path.mkdir(parents=True, exist_ok=True)
                completed += 1
                self.progress_changed.emit(completed, total_steps, str(folder_path))

            if self.options.create_registers:
                admin_folder = project_folder / "00_Project_Admin_and_Index"
                admin_folder.mkdir(parents=True, exist_ok=True)

                for filename, headers in PROJECT_REGISTER_TEMPLATES.items():
                    write_csv_template(admin_folder / filename, headers)
                    completed += 1
                    self.progress_changed.emit(completed, total_steps, str(admin_folder / filename))

                for test in self.options.tests:
                    test_folder_name = sanitize_folder_name(test)
                    test_id = test_folder_name.split("_")[0] if "_" in test_folder_name else test_folder_name
                    test_index_folder = project_folder / "03_Tests" / test_folder_name / "00_Test_Index"
                    test_index_folder.mkdir(parents=True, exist_ok=True)

                    for filename_template, headers in TEST_REGISTER_TEMPLATES.items():
                        filename = filename_template.format(test_id=test_id)
                        write_csv_template(test_index_folder / filename, headers)
                        completed += 1
                        self.progress_changed.emit(completed, total_steps, str(test_index_folder / filename))

            if self.options.create_readme:
                readme_path = project_folder / "README.md"
                if not readme_path.exists():
                    readme_path.write_text(README_TEXT, encoding="utf-8")
                completed += 1
                self.progress_changed.emit(completed, total_steps, str(readme_path))

            self.finished_successfully.emit(str(project_folder))

        except Exception as exc:
            self.failed.emit(str(exc))


class ValidationFolderTemplateCreator(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.worker: FolderCreationWorker | None = None

        self.setWindowTitle("Validation Folder Template Creator")
        self.resize(980, 720)

        self.destination_edit = QLineEdit()
        self.project_name_edit = QLineEdit("Project_Validation")
        self.test_input = QPlainTextEdit()
        self.test_input.setPlainText(
            "T001_CompressorCasing_Static_Load_Test\n"
            "T002_CompressorCasing_Pressure_Test\n"
            "T003_CompressorCasing_Vibration_Test"
        )

        self.create_registers_checkbox = QCheckBox("Create CSV register templates")
        self.create_registers_checkbox.setChecked(True)

        self.create_readme_checkbox = QCheckBox("Create README.md")
        self.create_readme_checkbox.setChecked(True)

        self.preview_list = QListWidget()
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)

        self.create_button = QPushButton("Create Folder Template")
        self.preview_button = QPushButton("Refresh Preview")
        self.browse_button = QPushButton("Browse...")

        self._build_ui()
        self._connect_signals()
        self.refresh_preview()

    def _build_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)

        title = QLabel("Validation Folder Template Creator")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)

        subtitle = QLabel(
            "Create a structured folder system for multi-test structural validation projects."
        )
        subtitle.setWordWrap(True)

        root_layout.addWidget(title)
        root_layout.addWidget(subtitle)

        setup_group = QGroupBox("Project Setup")
        setup_layout = QGridLayout(setup_group)

        setup_layout.addWidget(QLabel("Destination folder:"), 0, 0)
        setup_layout.addWidget(self.destination_edit, 0, 1)
        setup_layout.addWidget(self.browse_button, 0, 2)

        setup_layout.addWidget(QLabel("Project folder name:"), 1, 0)
        setup_layout.addWidget(self.project_name_edit, 1, 1, 1, 2)

        setup_layout.addWidget(QLabel("Tests, one per line:"), 2, 0, alignment=Qt.AlignmentFlag.AlignTop)
        setup_layout.addWidget(self.test_input, 2, 1, 1, 2)

        setup_layout.addWidget(self.create_registers_checkbox, 3, 1)
        setup_layout.addWidget(self.create_readme_checkbox, 4, 1)

        root_layout.addWidget(setup_group)

        middle_layout = QHBoxLayout()

        preview_group = QGroupBox("Preview")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.addWidget(self.preview_list)
        preview_layout.addWidget(self.preview_button)

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        log_layout.addWidget(self.log_output)

        middle_layout.addWidget(preview_group, 1)
        middle_layout.addWidget(log_group, 1)

        root_layout.addLayout(middle_layout)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.progress_bar, 1)
        bottom_layout.addWidget(self.create_button)

        root_layout.addLayout(bottom_layout)

        self.test_input.setMinimumHeight(110)
        self.preview_list.setMinimumHeight(260)
        self.log_output.setMinimumHeight(260)
        self.destination_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def _connect_signals(self) -> None:
        self.browse_button.clicked.connect(self.browse_destination)
        self.preview_button.clicked.connect(self.refresh_preview)
        self.create_button.clicked.connect(self.create_template)
        self.project_name_edit.textChanged.connect(self.refresh_preview)
        self.test_input.textChanged.connect(self.refresh_preview)

    def browse_destination(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Destination Folder",
            str(Path.home()),
        )

        if folder:
            self.destination_edit.setText(folder)

    def get_tests(self) -> list[str]:
        tests: list[str] = []

        for line in self.test_input.toPlainText().splitlines():
            cleaned = sanitize_folder_name(line)
            if cleaned:
                tests.append(cleaned)

        return tests

    def refresh_preview(self) -> None:
        self.preview_list.clear()

        project_name = sanitize_folder_name(self.project_name_edit.text()) or "Project_Validation"
        tests = self.get_tests()

        self.preview_list.addItem(project_name + "/")

        for folder in PROJECT_FOLDERS:
            self.preview_list.addItem(f"  {folder}/")

        for test in tests:
            self.preview_list.addItem(f"  03_Tests/{test}/")
            for subfolder in TEST_SUBFOLDERS:
                self.preview_list.addItem(f"    {subfolder}/")

        if self.create_registers_checkbox.isChecked():
            self.preview_list.addItem("  00_Project_Admin_and_Index/*.csv")
            for test in tests:
                self.preview_list.addItem(f"  03_Tests/{test}/00_Test_Index/*.csv")

        if self.create_readme_checkbox.isChecked():
            self.preview_list.addItem("  README.md")

    def validate_inputs(self) -> CreationOptions | None:
        destination_text = self.destination_edit.text().strip()
        project_name = sanitize_folder_name(self.project_name_edit.text())
        tests = self.get_tests()

        if not destination_text:
            QMessageBox.warning(self, "Missing Destination", "Please select a destination folder.")
            return None

        destination = Path(destination_text)

        if not destination.exists():
            QMessageBox.warning(self, "Invalid Destination", "The selected destination folder does not exist.")
            return None

        if not project_name:
            QMessageBox.warning(self, "Missing Project Name", "Please enter a project folder name.")
            return None

        if not tests:
            QMessageBox.warning(self, "Missing Tests", "Please enter at least one test name.")
            return None

        return CreationOptions(
            destination=destination,
            project_name=project_name,
            tests=tests,
            create_registers=self.create_registers_checkbox.isChecked(),
            create_readme=self.create_readme_checkbox.isChecked(),
        )

    def create_template(self) -> None:
        options = self.validate_inputs()

        if options is None:
            return

        project_folder = options.destination / sanitize_folder_name(options.project_name)

        if project_folder.exists():
            reply = QMessageBox.question(
                self,
                "Folder Already Exists",
                (
                    f"The folder already exists:\n\n{project_folder}\n\n"
                    "Existing files will not be overwritten, but missing folders/templates may be added.\n\n"
                    "Continue?"
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if reply != QMessageBox.StandardButton.Yes:
                return

        self.set_controls_enabled(False)
        self.progress_bar.setValue(0)
        self.log_output.clear()
        self.log_output.append("Creating folder template...")

        self.worker = FolderCreationWorker(options)
        self.worker.progress_changed.connect(self.on_progress_changed)
        self.worker.finished_successfully.connect(self.on_finished_successfully)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def set_controls_enabled(self, enabled: bool) -> None:
        self.destination_edit.setEnabled(enabled)
        self.project_name_edit.setEnabled(enabled)
        self.test_input.setEnabled(enabled)
        self.create_registers_checkbox.setEnabled(enabled)
        self.create_readme_checkbox.setEnabled(enabled)
        self.browse_button.setEnabled(enabled)
        self.preview_button.setEnabled(enabled)
        self.create_button.setEnabled(enabled)

    def on_progress_changed(self, completed: int, total: int, item: str) -> None:
        percent = int((completed / total) * 100) if total else 0
        self.progress_bar.setValue(percent)
        self.log_output.append(f"[{completed}/{total}] {item}")

    def on_finished_successfully(self, project_folder: str) -> None:
        self.set_controls_enabled(True)
        self.progress_bar.setValue(100)
        self.log_output.append("")
        self.log_output.append(f"Completed: {project_folder}")
        QMessageBox.information(
            self,
            "Completed",
            f"Folder template created successfully:\n\n{project_folder}",
        )

    def on_failed(self, error_message: str) -> None:
        self.set_controls_enabled(True)
        self.log_output.append("")
        self.log_output.append(f"Error: {error_message}")
        QMessageBox.critical(self, "Error", error_message)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = ValidationFolderTemplateCreator()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
