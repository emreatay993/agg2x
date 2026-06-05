"""
Validation Folder Tree Indexer

A PyQt6 GUI utility for creating a recursive file/folder tree under a selected folder.
It can exclude user-specified file extensions to avoid bloated indexes from solver
intermediate files, temporary files, and unnecessary outputs.

Install dependency:
    pip install PyQt6

Run:
    python validation_tree_indexer.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

from PyQt6.QtCore import QThread, Qt, pyqtSignal
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
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


DEFAULT_EXCLUDED_EXTENSIONS = (
    ".rst, .rth, .rdb, .db, .esav, .emat, .full, .mode, .mntr, "
    ".err, .out, .log, .tmp, .bak, .lock, .mechdb"
)


@dataclass
class TreeStats:
    included_folders: int = 0
    included_files: int = 0
    excluded_files: int = 0
    inaccessible_items: int = 0
    excluded_by_extension: dict[str, int] = field(default_factory=dict)


@dataclass
class TreeOptions:
    root_folder: Path
    excluded_extensions: set[str]
    max_depth_enabled: bool
    max_depth: int
    ignore_hidden: bool
    include_file_size: bool
    include_modified_date: bool
    folders_first: bool
    show_exclusion_summary: bool
    markdown_output: bool


def normalize_extensions(text: str) -> set[str]:
    raw_parts = text.replace(";", ",").replace("\n", ",").replace("\t", ",").split(",")
    extensions: set[str] = set()

    for part in raw_parts:
        cleaned = part.strip().lower()

        if not cleaned:
            continue

        if cleaned == "*":
            continue

        if not cleaned.startswith("."):
            cleaned = "." + cleaned

        extensions.add(cleaned)

    return extensions


def is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]

    value = float(num_bytes)

    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024

    return f"{num_bytes} B"


def format_metadata(path: Path, include_file_size: bool, include_modified_date: bool) -> str:
    metadata_parts: list[str] = []

    try:
        stat = path.stat()
    except OSError:
        return ""

    if include_file_size and path.is_file():
        metadata_parts.append(format_size(stat.st_size))

    if include_modified_date:
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        metadata_parts.append(f"modified {modified}")

    if not metadata_parts:
        return ""

    return "  [" + " | ".join(metadata_parts) + "]"


class TreeBuilder:
    def __init__(self, options: TreeOptions) -> None:
        self.options = options
        self.stats = TreeStats()
        self.cancel_requested = False

    def cancel(self) -> None:
        self.cancel_requested = True

    def build(self) -> tuple[str, TreeStats]:
        root = self.options.root_folder
        lines: list[str] = []

        title = root.name + "/"
        title += format_metadata(root, False, self.options.include_modified_date)
        lines.append(title)

        self.stats.included_folders += 1
        self._append_children(root, lines, prefix="", depth=0)

        if self.options.show_exclusion_summary:
            lines.append("")
            lines.append("Summary")
            lines.append("-------")
            lines.append(f"Included folders: {self.stats.included_folders}")
            lines.append(f"Included files: {self.stats.included_files}")
            lines.append(f"Excluded files: {self.stats.excluded_files}")
            lines.append(f"Inaccessible items: {self.stats.inaccessible_items}")

            if self.stats.excluded_by_extension:
                lines.append("")
                lines.append("Excluded by extension:")
                for extension, count in sorted(self.stats.excluded_by_extension.items()):
                    lines.append(f"- {extension}: {count}")

        tree_text = "\n".join(lines)

        if self.options.markdown_output:
            tree_text = f"# Folder Tree Index\n\nRoot folder: `{root}`\n\n```text\n{tree_text}\n```\n"

        return tree_text, self.stats

    def _append_children(self, folder: Path, lines: list[str], prefix: str, depth: int) -> None:
        if self.cancel_requested:
            return

        if self.options.max_depth_enabled and depth >= self.options.max_depth:
            return

        try:
            children = list(folder.iterdir())
        except OSError:
            self.stats.inaccessible_items += 1
            lines.append(prefix + "└── " + "[inaccessible]")
            return

        if self.options.ignore_hidden:
            children = [child for child in children if not is_hidden(child.relative_to(self.options.root_folder))]

        visible_children: list[Path] = []

        for child in children:
            if child.is_file() and child.suffix.lower() in self.options.excluded_extensions:
                extension = child.suffix.lower()
                self.stats.excluded_files += 1
                self.stats.excluded_by_extension[extension] = self.stats.excluded_by_extension.get(extension, 0) + 1
                continue

            visible_children.append(child)

        if self.options.folders_first:
            visible_children.sort(key=lambda p: (not p.is_dir(), p.name.lower()))
        else:
            visible_children.sort(key=lambda p: p.name.lower())

        for index, child in enumerate(visible_children):
            if self.cancel_requested:
                return

            is_last = index == len(visible_children) - 1
            connector = "└── " if is_last else "├── "
            child_prefix = prefix + connector
            next_prefix = prefix + ("    " if is_last else "│   ")

            try:
                if child.is_dir():
                    name = child.name + "/"
                    metadata = format_metadata(child, False, self.options.include_modified_date)
                    lines.append(child_prefix + name + metadata)
                    self.stats.included_folders += 1
                    self._append_children(child, lines, next_prefix, depth + 1)
                elif child.is_file():
                    metadata = format_metadata(
                        child,
                        self.options.include_file_size,
                        self.options.include_modified_date,
                    )
                    lines.append(child_prefix + child.name + metadata)
                    self.stats.included_files += 1
                else:
                    lines.append(child_prefix + child.name)
            except OSError:
                self.stats.inaccessible_items += 1
                lines.append(child_prefix + child.name + " [inaccessible]")


class TreeWorker(QThread):
    progress_started = pyqtSignal()
    finished_successfully = pyqtSignal(str, TreeStats)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, options: TreeOptions) -> None:
        super().__init__()
        self.builder = TreeBuilder(options)

    def cancel(self) -> None:
        self.builder.cancel()

    def run(self) -> None:
        try:
            self.progress_started.emit()
            tree_text, stats = self.builder.build()

            if self.builder.cancel_requested:
                self.cancelled.emit()
                return

            self.finished_successfully.emit(tree_text, stats)

        except Exception as exc:
            self.failed.emit(str(exc))


class ValidationTreeIndexer(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.worker: TreeWorker | None = None
        self.current_tree_text = ""

        self.setWindowTitle("Validation Folder Tree Indexer")
        self.resize(1100, 760)

        self.folder_edit = QLineEdit()
        self.excluded_extensions_edit = QPlainTextEdit()
        self.excluded_extensions_edit.setPlainText(DEFAULT_EXCLUDED_EXTENSIONS)

        self.ignore_hidden_checkbox = QCheckBox("Ignore hidden files/folders")
        self.ignore_hidden_checkbox.setChecked(True)

        self.include_file_size_checkbox = QCheckBox("Include file sizes")
        self.include_file_size_checkbox.setChecked(True)

        self.include_modified_date_checkbox = QCheckBox("Include modified dates")
        self.include_modified_date_checkbox.setChecked(False)

        self.folders_first_checkbox = QCheckBox("List folders before files")
        self.folders_first_checkbox.setChecked(True)

        self.show_summary_checkbox = QCheckBox("Add exclusion/statistics summary")
        self.show_summary_checkbox.setChecked(True)

        self.markdown_output_checkbox = QCheckBox("Format output as Markdown")
        self.markdown_output_checkbox.setChecked(True)

        self.max_depth_checkbox = QCheckBox("Limit scan depth")
        self.max_depth_checkbox.setChecked(False)

        self.max_depth_spinbox = QSpinBox()
        self.max_depth_spinbox.setRange(1, 99)
        self.max_depth_spinbox.setValue(8)
        self.max_depth_spinbox.setEnabled(False)

        self.browse_button = QPushButton("Browse...")
        self.generate_button = QPushButton("Generate Tree")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)

        self.save_button = QPushButton("Save Output...")
        self.save_button.setEnabled(False)

        self.copy_button = QPushButton("Copy to Clipboard")
        self.copy_button.setEnabled(False)

        self.clear_button = QPushButton("Clear Output")

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        self.output_text.setFont(QFont("Consolas", 10))

        self.stats_label = QLabel("No tree generated yet.")
        self.stats_label.setWordWrap(True)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)

        self._build_ui()
        self._connect_signals()

    def _build_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        root_layout = QVBoxLayout(central_widget)

        title = QLabel("Validation Folder Tree Indexer")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title.setFont(title_font)

        subtitle = QLabel(
            "Create a recursive folder/file index while excluding unnecessary extensions such as "
            "solver intermediates, logs, temporary files, and large output artifacts."
        )
        subtitle.setWordWrap(True)

        root_layout.addWidget(title)
        root_layout.addWidget(subtitle)

        setup_group = QGroupBox("Scan Setup")
        setup_layout = QGridLayout(setup_group)

        setup_layout.addWidget(QLabel("Root folder:"), 0, 0)
        setup_layout.addWidget(self.folder_edit, 0, 1)
        setup_layout.addWidget(self.browse_button, 0, 2)

        setup_layout.addWidget(QLabel("Excluded extensions:"), 1, 0, alignment=Qt.AlignmentFlag.AlignTop)
        setup_layout.addWidget(self.excluded_extensions_edit, 1, 1, 1, 2)

        depth_layout = QHBoxLayout()
        depth_layout.addWidget(self.max_depth_checkbox)
        depth_layout.addWidget(self.max_depth_spinbox)
        depth_layout.addStretch()

        setup_layout.addLayout(depth_layout, 2, 1, 1, 2)

        option_layout_1 = QHBoxLayout()
        option_layout_1.addWidget(self.ignore_hidden_checkbox)
        option_layout_1.addWidget(self.include_file_size_checkbox)
        option_layout_1.addWidget(self.include_modified_date_checkbox)
        option_layout_1.addStretch()

        option_layout_2 = QHBoxLayout()
        option_layout_2.addWidget(self.folders_first_checkbox)
        option_layout_2.addWidget(self.show_summary_checkbox)
        option_layout_2.addWidget(self.markdown_output_checkbox)
        option_layout_2.addStretch()

        setup_layout.addLayout(option_layout_1, 3, 1, 1, 2)
        setup_layout.addLayout(option_layout_2, 4, 1, 1, 2)

        root_layout.addWidget(setup_group)

        output_group = QGroupBox("Tree Output")
        output_layout = QVBoxLayout(output_group)
        output_layout.addWidget(self.output_text)

        root_layout.addWidget(output_group, 1)

        bottom_layout = QVBoxLayout()

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.generate_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addStretch()
        button_layout.addWidget(self.copy_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.clear_button)

        bottom_layout.addWidget(self.stats_label)
        bottom_layout.addWidget(self.progress_bar)
        bottom_layout.addLayout(button_layout)

        root_layout.addLayout(bottom_layout)

        self.folder_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.excluded_extensions_edit.setMaximumHeight(85)

    def _connect_signals(self) -> None:
        self.browse_button.clicked.connect(self.browse_folder)
        self.generate_button.clicked.connect(self.generate_tree)
        self.cancel_button.clicked.connect(self.cancel_scan)
        self.save_button.clicked.connect(self.save_output)
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        self.clear_button.clicked.connect(self.clear_output)
        self.max_depth_checkbox.toggled.connect(self.max_depth_spinbox.setEnabled)

    def browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Root Folder",
            str(Path.home()),
        )

        if folder:
            self.folder_edit.setText(folder)

    def validate_options(self) -> TreeOptions | None:
        root_text = self.folder_edit.text().strip()

        if not root_text:
            QMessageBox.warning(self, "Missing Folder", "Please select a root folder.")
            return None

        root_folder = Path(root_text)

        if not root_folder.exists() or not root_folder.is_dir():
            QMessageBox.warning(self, "Invalid Folder", "The selected root folder does not exist.")
            return None

        excluded_extensions = normalize_extensions(self.excluded_extensions_edit.toPlainText())

        return TreeOptions(
            root_folder=root_folder,
            excluded_extensions=excluded_extensions,
            max_depth_enabled=self.max_depth_checkbox.isChecked(),
            max_depth=self.max_depth_spinbox.value(),
            ignore_hidden=self.ignore_hidden_checkbox.isChecked(),
            include_file_size=self.include_file_size_checkbox.isChecked(),
            include_modified_date=self.include_modified_date_checkbox.isChecked(),
            folders_first=self.folders_first_checkbox.isChecked(),
            show_exclusion_summary=self.show_summary_checkbox.isChecked(),
            markdown_output=self.markdown_output_checkbox.isChecked(),
        )

    def set_controls_enabled(self, enabled: bool) -> None:
        self.folder_edit.setEnabled(enabled)
        self.excluded_extensions_edit.setEnabled(enabled)
        self.ignore_hidden_checkbox.setEnabled(enabled)
        self.include_file_size_checkbox.setEnabled(enabled)
        self.include_modified_date_checkbox.setEnabled(enabled)
        self.folders_first_checkbox.setEnabled(enabled)
        self.show_summary_checkbox.setEnabled(enabled)
        self.markdown_output_checkbox.setEnabled(enabled)
        self.max_depth_checkbox.setEnabled(enabled)
        self.max_depth_spinbox.setEnabled(enabled and self.max_depth_checkbox.isChecked())
        self.browse_button.setEnabled(enabled)
        self.generate_button.setEnabled(enabled)
        self.cancel_button.setEnabled(not enabled)

    def generate_tree(self) -> None:
        options = self.validate_options()

        if options is None:
            return

        self.current_tree_text = ""
        self.output_text.clear()
        self.stats_label.setText("Scanning...")
        self.progress_bar.setRange(0, 0)
        self.save_button.setEnabled(False)
        self.copy_button.setEnabled(False)
        self.set_controls_enabled(False)

        self.worker = TreeWorker(options)
        self.worker.finished_successfully.connect(self.on_finished_successfully)
        self.worker.failed.connect(self.on_failed)
        self.worker.cancelled.connect(self.on_cancelled)
        self.worker.start()

    def cancel_scan(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.stats_label.setText("Cancelling...")

    def on_finished_successfully(self, tree_text: str, stats: TreeStats) -> None:
        self.current_tree_text = tree_text
        self.output_text.setPlainText(tree_text)

        self.stats_label.setText(
            f"Included folders: {stats.included_folders} | "
            f"Included files: {stats.included_files} | "
            f"Excluded files: {stats.excluded_files} | "
            f"Inaccessible items: {stats.inaccessible_items}"
        )

        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.save_button.setEnabled(True)
        self.copy_button.setEnabled(True)
        self.set_controls_enabled(True)

    def on_failed(self, error_message: str) -> None:
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.stats_label.setText("Scan failed.")
        self.set_controls_enabled(True)
        QMessageBox.critical(self, "Error", error_message)

    def on_cancelled(self) -> None:
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.stats_label.setText("Scan cancelled.")
        self.set_controls_enabled(True)

    def save_output(self) -> None:
        if not self.current_tree_text:
            QMessageBox.information(self, "No Output", "Generate a tree before saving.")
            return

        default_extension = "md" if self.markdown_output_checkbox.isChecked() else "txt"
        default_filename = f"folder_tree_index.{default_extension}"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Folder Tree",
            str(Path.home() / default_filename),
            "Markdown Files (*.md);;Text Files (*.txt);;All Files (*)",
        )

        if not file_path:
            return

        Path(file_path).write_text(self.current_tree_text, encoding="utf-8")
        QMessageBox.information(self, "Saved", f"Tree index saved:\n\n{file_path}")

    def copy_to_clipboard(self) -> None:
        if not self.current_tree_text:
            QMessageBox.information(self, "No Output", "Generate a tree before copying.")
            return

        QApplication.clipboard().setText(self.current_tree_text)
        self.stats_label.setText("Tree output copied to clipboard.")

    def clear_output(self) -> None:
        self.current_tree_text = ""
        self.output_text.clear()
        self.stats_label.setText("No tree generated yet.")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.save_button.setEnabled(False)
        self.copy_button.setEnabled(False)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    window = ValidationTreeIndexer()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
