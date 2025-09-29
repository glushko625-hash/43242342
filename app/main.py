"""Desktop note editor with Gemini-powered contextual editing."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QAction, QCloseEvent, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ai import GeminiClient, GeminiError
from note_store import Note, NoteStore

APP_NAME = "Aurora Notes"


class AiWorker(QThread):
    """Runs Gemini requests off the UI thread."""

    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, client: GeminiClient, note: str, selection: str, instruction: str) -> None:
        super().__init__()
        self._client = client
        self._note = note
        self._selection = selection
        self._instruction = instruction

    def run(self) -> None:  # noqa: D401 - QThread.run signature
        try:
            result = self._client.transform_selection(self._note, self._selection, self._instruction)
        except GeminiError as exc:
            self.failed.emit(str(exc))
            return
        self.completed.emit(result)


class MainWindow(QMainWindow):
    """Main application window with note list and editor."""

    def __init__(self, store: NoteStore, ai_client: Optional[GeminiClient]) -> None:
        super().__init__()
        self.store = store
        self.ai_client = ai_client
        self.active_note: Optional[Note] = None
        self.ai_thread: Optional[AiWorker] = None

        self.setWindowTitle(APP_NAME)
        self.resize(1200, 720)
        self._build_ui()
        self._load_notes()

    # region UI setup -----------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left panel: notes list + actions
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        header = QLabel("Заметки")
        header.setObjectName("NotesHeader")
        left_layout.addWidget(header)

        self.note_list = QListWidget()
        self.note_list.setObjectName("NotesList")
        self.note_list.itemSelectionChanged.connect(self._handle_note_selection)
        left_layout.addWidget(self.note_list)

        note_buttons = QWidget()
        note_buttons_layout = QHBoxLayout(note_buttons)
        note_buttons_layout.setContentsMargins(0, 0, 0, 0)
        note_buttons_layout.setSpacing(8)

        self.new_button = QPushButton("Новая")
        self.new_button.clicked.connect(self._create_note)
        note_buttons_layout.addWidget(self.new_button)

        self.rename_button = QPushButton("Переименовать")
        self.rename_button.clicked.connect(self._rename_note)
        note_buttons_layout.addWidget(self.rename_button)

        left_layout.addWidget(note_buttons)

        # Right panel: editor + AI tools
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        toolbar = QToolBar()
        toolbar.setIconSize(toolbar.iconSize() * 1.5)
        right_layout.addWidget(toolbar)

        self.ai_button = QAction("AI: изменить выделение", self)
        self.ai_button.triggered.connect(self._ask_ai_to_edit)
        self.ai_button.setEnabled(self.ai_client is not None)
        toolbar.addAction(self.ai_button)

        export_action = QAction("Экспортировать…", self)
        export_action.triggered.connect(self._export_note)
        toolbar.addAction(export_action)

        self.editor = QTextEdit()
        self.editor.setObjectName("Editor")
        self.editor.textChanged.connect(self._persist_active_note)
        right_layout.addWidget(self.editor)

        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        self._apply_theme()

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #10121a;
            }
            QWidget {
                font-family: 'Segoe UI', 'Inter', sans-serif;
                color: #f4f6fb;
                background-color: transparent;
            }
            QListWidget#NotesList {
                background-color: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 12px;
                padding: 8px;
            }
            QListWidget#NotesList::item {
                padding: 12px 10px;
                border-radius: 10px;
            }
            QListWidget#NotesList::item:selected {
                background-color: #4f46e5;
            }
            QLabel#NotesHeader {
                font-size: 18px;
                font-weight: 600;
                margin-bottom: 12px;
            }
            QPushButton {
                background-color: #1f2937;
                border-radius: 10px;
                padding: 8px 12px;
            }
            QPushButton:hover {
                background-color: #374151;
            }
            QPushButton:pressed {
                background-color: #4f46e5;
            }
            QTextEdit#Editor {
                background-color: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 16px;
                padding: 18px;
                font-size: 15px;
                line-height: 1.5em;
            }
            QToolBar {
                background-color: transparent;
                spacing: 12px;
                padding: 8px 0 16px 0;
            }
            QToolBar QToolButton {
                background-color: #1f2937;
                border-radius: 12px;
                padding: 10px 14px;
            }
            QToolBar QToolButton:hover {
                background-color: #4f46e5;
            }
        """
        )

    # endregion ----------------------------------------------------------

    def _load_notes(self) -> None:
        notes = self.store.list_notes()
        if not notes:
            sample = self.store.create_note("Первая заметка")
            sample.content = (
                "# Добро пожаловать в Aurora Notes\n\n"
                "Выделите текст, нажмите на кнопку AI и опишите, что нужно сделать."
            )
            self.store.save_note(sample)
            notes = [sample]

        for note in notes:
            item = QListWidgetItem(note.title)
            item.setData(Qt.UserRole, note)
            self.note_list.addItem(item)

        if self.note_list.count():
            self.note_list.setCurrentRow(0)

    def _persist_active_note(self) -> None:
        if not self.active_note:
            return
        self.active_note.content = self.editor.toPlainText()
        self.store.save_note(self.active_note)

    def _handle_note_selection(self) -> None:
        items = self.note_list.selectedItems()
        if not items:
            self.editor.clear()
            self.active_note = None
            return
        item = items[0]
        note: Note = item.data(Qt.UserRole)
        self.active_note = note
        self.editor.blockSignals(True)
        self.editor.setPlainText(note.content)
        self.editor.blockSignals(False)

    def _create_note(self) -> None:
        title, ok = QInputDialog.getText(self, "Новая заметка", "Название:", text="Новая заметка")
        if not ok or not title.strip():
            return
        note = self.store.create_note(title.strip())
        item = QListWidgetItem(note.title)
        item.setData(Qt.UserRole, note)
        self.note_list.addItem(item)
        self.note_list.setCurrentItem(item)

    def _rename_note(self) -> None:
        if not self.active_note:
            return
        title, ok = QInputDialog.getText(self, "Переименовать", "Новое название:", text=self.active_note.title)
        if not ok or not title.strip():
            return
        self.active_note.title = title.strip()
        self.store.save_note(self.active_note)
        item = self.note_list.currentItem()
        if item:
            item.setText(self.active_note.title)

    def _export_note(self) -> None:
        if not self.active_note:
            return
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Экспортировать заметку",
            self.active_note.title + ".md",
            "Markdown (*.md);;Text (*.txt)",
        )
        if not filename:
            return
        Path(filename).write_text(self.editor.toPlainText(), encoding="utf-8")
        QMessageBox.information(self, "Готово", "Заметка экспортирована.")

    def _ask_ai_to_edit(self) -> None:
        if not self.ai_client:
            QMessageBox.warning(self, "AI отключен", "Настройте GEMINI_API_KEY, чтобы использовать ИИ.")
            return
        selection = self.editor.textCursor().selectedText()
        if not selection:
            QMessageBox.warning(self, "Нет выделения", "Выделите текст для изменения.")
            return
        instruction, ok = QInputDialog.getText(self, "AI помощник", "Что нужно сделать с выделенным текстом?")
        if not ok or not instruction.strip():
            return
        self.statusBar().showMessage("ИИ думает…", 0)
        self.ai_button.setEnabled(False)
        self.ai_thread = AiWorker(
            self.ai_client,
            self.editor.toPlainText(),
            selection,
            instruction.strip(),
        )
        self.ai_thread.completed.connect(self._apply_ai_result)
        self.ai_thread.failed.connect(self._ai_failed)
        self.ai_thread.finished.connect(self._cleanup_ai_thread)
        self.ai_thread.start()

    def _apply_ai_result(self, replacement: str) -> None:
        cursor = self.editor.textCursor()
        cursor.beginEditBlock()
        cursor.removeSelectedText()
        cursor.insertText(replacement)
        cursor.endEditBlock()
        self.statusBar().showMessage("Готово", 3000)
        self._persist_active_note()

    def _ai_failed(self, error: str) -> None:
        QMessageBox.critical(self, "Ошибка ИИ", error)
        self.statusBar().clearMessage()

    def _cleanup_ai_thread(self) -> None:
        self.ai_button.setEnabled(True)
        self.ai_thread = None

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: D401 - Qt signature
        if self.ai_thread and self.ai_thread.isRunning():
            QMessageBox.warning(
                self,
                "ИИ занят",
                "Дождитесь завершения запроса к ИИ перед закрытием приложения.",
            )
            event.ignore()
            return
        super().closeEvent(event)


def ensure_store() -> NoteStore:
    data_dir = Path.home() / ".aurora_notes"
    return NoteStore(data_dir)


def build_ai_client() -> Optional[GeminiClient]:
    try:
        return GeminiClient()
    except GeminiError:
        return None


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(QIcon())

    store = ensure_store()
    ai_client = build_ai_client()

    window = MainWindow(store, ai_client)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
