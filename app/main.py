"""Desktop note editor with Gemini-powered contextual editing."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QCloseEvent, QIcon, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ai import GeminiClient, GeminiConfig, GeminiError, available_model_specs
from note_store import Note, NoteStore
from settings_store import AppSettings, SettingsStore

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


class ChatWorker(QThread):
    """Runs chat completions off the UI thread."""

    completed = Signal(str)
    failed = Signal(str)

    def __init__(self, client: GeminiClient, note: str, history: List[Dict[str, str]]) -> None:
        super().__init__()
        self._client = client
        self._note = note
        self._history = history

    def run(self) -> None:  # noqa: D401 - QThread.run signature
        try:
            result = self._client.chat(self._note, self._history)
        except GeminiError as exc:
            self.failed.emit(str(exc))
            return
        self.completed.emit(result)


class SettingsDialog(QDialog):
    """Dialog that lets the user configure Gemini credentials."""

    def __init__(self, parent: QWidget, settings: AppSettings) -> None:
        super().__init__(parent)
        self.setWindowTitle("Настройки ИИ")
        self.setModal(True)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.api_key_edit = QLineEdit(settings.api_key)
        self.api_key_edit.setEchoMode(QLineEdit.Password)
        self.api_key_edit.setPlaceholderText("Введите GEMINI_API_KEY")
        form.addRow("API ключ", self.api_key_edit)

        self._model = settings.model
        model_hint = QLabel(
            "Текущая модель настраивается прямо в панели чата."
        )
        model_hint.setWordWrap(True)
        form.addRow("Модель", model_hint)

        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> AppSettings:
        return AppSettings(api_key=self.api_key_edit.text().strip(), model=self._model)


class MainWindow(QMainWindow):
    """Main application window with note list, editor, and AI helpers."""

    def __init__(
        self,
        store: NoteStore,
        settings_store: SettingsStore,
        initial_settings: AppSettings,
    ) -> None:
        super().__init__()
        self.store = store
        self.settings_store = settings_store
        self.settings = initial_settings
        self.ai_client: Optional[GeminiClient] = None
        self.active_note: Optional[Note] = None
        self.ai_thread: Optional[AiWorker] = None
        self.chat_thread: Optional[ChatWorker] = None
        self.pending_ai_range: Optional[tuple[int, int]] = None
        self.chat_history: List[Dict[str, str]] = []
        self.model_combo: Optional[QComboBox] = None

        self.setWindowTitle(APP_NAME)
        self.resize(1200, 720)
        self._build_ui()
        self._load_notes()
        self._populate_model_selector()
        self._apply_settings(update_status=False)

    # region UI setup -----------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)

        layout = QHBoxLayout(central)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        layout.addWidget(splitter)

        # Left panel: notes list + actions
        left_panel = QFrame()
        left_panel.setObjectName("NotesPanel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(16, 16, 16, 16)
        left_layout.setSpacing(12)

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
        self.new_button.setObjectName("NewNoteButton")
        self.new_button.clicked.connect(self._create_note)
        note_buttons_layout.addWidget(self.new_button)

        self.rename_button = QPushButton("Переименовать")
        self.rename_button.setObjectName("RenameNoteButton")
        self.rename_button.clicked.connect(self._rename_note)
        note_buttons_layout.addWidget(self.rename_button)

        left_layout.addWidget(note_buttons)

        # Center panel: editor and actions
        editor_card = QFrame()
        editor_card.setObjectName("EditorCard")
        editor_layout = QVBoxLayout(editor_card)
        editor_layout.setContentsMargins(24, 24, 24, 24)
        editor_layout.setSpacing(16)

        editor_header = QWidget()
        editor_header_layout = QHBoxLayout(editor_header)
        editor_header_layout.setContentsMargins(0, 0, 0, 0)
        editor_header_layout.setSpacing(12)

        editor_title = QLabel("Редактор заметки")
        editor_title.setObjectName("EditorHeader")
        editor_header_layout.addWidget(editor_title)
        editor_header_layout.addStretch(1)

        self.ai_button = QPushButton("Изменить выделение")
        self.ai_button.setObjectName("AiButton")
        self.ai_button.clicked.connect(self._ask_ai_to_edit)
        editor_header_layout.addWidget(self.ai_button)

        self.export_button = QPushButton("Экспорт…")
        self.export_button.setObjectName("ExportButton")
        self.export_button.clicked.connect(self._export_note)
        editor_header_layout.addWidget(self.export_button)

        self.settings_button = QPushButton("Настройки")
        self.settings_button.setObjectName("SettingsButton")
        self.settings_button.clicked.connect(self._open_settings)
        editor_header_layout.addWidget(self.settings_button)

        editor_layout.addWidget(editor_header)

        self.editor = QTextEdit()
        self.editor.setObjectName("Editor")
        self.editor.textChanged.connect(self._persist_active_note)
        editor_layout.addWidget(self.editor, 1)

        # Right panel: chat and model selector
        chat_card = QFrame()
        chat_card.setObjectName("ChatCard")
        chat_layout = QVBoxLayout(chat_card)
        chat_layout.setContentsMargins(24, 24, 24, 24)
        chat_layout.setSpacing(16)

        chat_header_row = QWidget()
        chat_header_layout = QHBoxLayout(chat_header_row)
        chat_header_layout.setContentsMargins(0, 0, 0, 0)
        chat_header_layout.setSpacing(12)

        chat_header = QLabel("Чат с ИИ")
        chat_header.setObjectName("ChatHeader")
        chat_header_layout.addWidget(chat_header)
        chat_header_layout.addStretch(1)

        self.model_combo = QComboBox()
        self.model_combo.setObjectName("ModelSelector")
        self.model_combo.currentIndexChanged.connect(self._handle_model_change)
        chat_header_layout.addWidget(self.model_combo)

        chat_layout.addWidget(chat_header_row)

        self.chat_view = QTextEdit()
        self.chat_view.setObjectName("ChatView")
        self.chat_view.setReadOnly(True)
        chat_layout.addWidget(self.chat_view, 1)

        clear_button = QPushButton("Очистить чат")
        clear_button.setObjectName("ClearChatButton")
        clear_button.clicked.connect(self._clear_chat)
        chat_layout.addWidget(clear_button)

        input_row = QWidget()
        input_layout = QHBoxLayout(input_row)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(12)

        self.chat_input = QPlainTextEdit()
        self.chat_input.setObjectName("ChatInput")
        self.chat_input.setPlaceholderText("Спросите ИИ о заметке…")
        self.chat_input.setFixedHeight(100)
        input_layout.addWidget(self.chat_input, 1)

        self.chat_send_button = QPushButton("Отправить")
        self.chat_send_button.setObjectName("SendButton")
        self.chat_send_button.clicked.connect(self._send_chat_message)
        input_layout.addWidget(self.chat_send_button)

        chat_layout.addWidget(input_row)

        splitter.addWidget(left_panel)
        splitter.addWidget(editor_card)
        splitter.addWidget(chat_card)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)
        splitter.setStretchFactor(2, 1)

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
            QFrame#NotesPanel, QFrame#EditorCard, QFrame#ChatCard {
                background-color: rgba(255, 255, 255, 0.02);
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-radius: 24px;
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
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #374151;
            }
            QPushButton:pressed {
                background-color: #4f46e5;
            }
            QPushButton#AiButton {
                background-color: #4f46e5;
            }
            QPushButton#AiButton:hover {
                background-color: #6366f1;
            }
            QPushButton#SendButton {
                background-color: #2563eb;
            }
            QPushButton#SendButton:hover {
                background-color: #3b82f6;
            }
            QTextEdit#Editor {
                background-color: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 16px;
                padding: 18px;
                font-size: 15px;
                line-height: 1.5em;
            }
            QTextEdit#ChatView {
                background-color: rgba(255, 255, 255, 0.02);
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 14px;
                padding: 12px;
                font-size: 14px;
                line-height: 1.45em;
            }
            QPlainTextEdit#ChatInput {
                background-color: rgba(255, 255, 255, 0.02);
                border: 1px solid rgba(255, 255, 255, 0.06);
                border-radius: 12px;
                padding: 10px;
                font-size: 14px;
            }
            QLabel#ChatHeader {
                font-size: 16px;
                font-weight: 600;
            }
            QComboBox#ModelSelector {
                background-color: rgba(31, 41, 55, 0.8);
                border-radius: 10px;
                padding: 6px 12px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
            QComboBox#ModelSelector::drop-down {
                border: none;
            }
            QComboBox#ModelSelector QAbstractItemView {
                background-color: #111827;
                border-radius: 10px;
                selection-background-color: #4f46e5;
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
            self._clear_chat(silent=True)
            return
        item = items[0]
        note: Note = item.data(Qt.UserRole)
        self.active_note = note
        self.editor.blockSignals(True)
        self.editor.setPlainText(note.content)
        self.editor.blockSignals(False)
        self._clear_chat(silent=True)

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
        cursor = self.editor.textCursor()
        selection = cursor.selectedText().replace("\u2029", "\n")
        if not selection:
            QMessageBox.warning(self, "Нет выделения", "Выделите текст для изменения.")
            return
        instruction, ok = QInputDialog.getText(self, "AI помощник", "Что нужно сделать с выделенным текстом?")
        if not ok or not instruction.strip():
            return
        self.pending_ai_range = (cursor.selectionStart(), cursor.selectionEnd())
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
        if not self.pending_ai_range:
            return
        start, end = self.pending_ai_range
        cursor = self.editor.textCursor()
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.KeepAnchor)
        cursor.beginEditBlock()
        cursor.removeSelectedText()
        cursor.insertText(replacement)
        cursor.endEditBlock()
        self.statusBar().showMessage("Готово", 3000)
        self._persist_active_note()
        self.pending_ai_range = None

    def _ai_failed(self, error: str) -> None:
        QMessageBox.critical(self, "Ошибка ИИ", error)
        self.statusBar().clearMessage()
        self.pending_ai_range = None

    def _cleanup_ai_thread(self) -> None:
        self.ai_button.setEnabled(self.ai_client is not None)
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
        if self.chat_thread and self.chat_thread.isRunning():
            QMessageBox.warning(
                self,
                "Чат активен",
                "Дождитесь ответа ИИ перед закрытием приложения.",
            )
            event.ignore()
            return
        super().closeEvent(event)

    # region AI helpers -------------------------------------------------

    def _apply_settings(self, update_status: bool = True) -> None:
        api_key = self.settings.api_key or os.environ.get("GEMINI_API_KEY", "")
        try:
            if api_key:
                self.ai_client = GeminiClient(
                    GeminiConfig(api_key=api_key, model=self.settings.model)
                )
            else:
                self.ai_client = None
        except GeminiError as exc:
            self.ai_client = None
            if update_status:
                self.statusBar().showMessage(str(exc), 5000)
        else:
            if update_status and self.ai_client:
                spec = self.ai_client.model_spec
                self.statusBar().showMessage(
                    f"ИИ готов к работе ({spec.label})", 3000
                )

        available = self.ai_client is not None
        if hasattr(self, "ai_button"):
            self.ai_button.setEnabled(available)
        if hasattr(self, "chat_send_button"):
            self.chat_send_button.setEnabled(available)
        if not available and hasattr(self, "chat_view") and update_status:
            self.chat_view.append("ИИ недоступен. Откройте настройки, чтобы указать ключ API.")
        self._populate_model_selector()

    def _open_settings(self) -> None:
        dialog = SettingsDialog(self, self.settings)
        if dialog.exec() != QDialog.Accepted:
            return
        self.settings = dialog.values()
        self.settings_store.save(self.settings)
        self._apply_settings()

    def _populate_model_selector(self) -> None:
        if not self.model_combo:
            return

        current_model = self.settings.model or "gemini-2.5-pro"
        specs = available_model_specs()

        self.model_combo.blockSignals(True)
        self.model_combo.clear()

        known_ids = []
        for spec in specs:
            index = self.model_combo.count()
            self.model_combo.addItem(spec.label, spec.model_id)
            self.model_combo.setItemData(index, spec.description, Qt.ToolTipRole)
            known_ids.append(spec.model_id)

        if current_model not in known_ids:
            index = self.model_combo.count()
            self.model_combo.addItem(current_model, current_model)
            self.model_combo.setItemData(
                index,
                "Пользовательская модель из настроек.",
                Qt.ToolTipRole,
            )

        index = self.model_combo.findData(current_model)
        if index < 0:
            index = 0
        self.model_combo.setCurrentIndex(index)
        self.model_combo.blockSignals(False)

    def _handle_model_change(self, index: int) -> None:
        if not self.model_combo:
            return
        model_id = self.model_combo.itemData(index)
        if not model_id or model_id == self.settings.model:
            return
        self.settings.model = model_id
        self.settings_store.save(self.settings)
        self._apply_settings()

    def _clear_chat(self, silent: bool = False) -> None:
        self.chat_history.clear()
        self.chat_view.clear()
        if not silent:
            self.statusBar().showMessage("История чата очищена", 3000)

    def _send_chat_message(self) -> None:
        if not self.ai_client:
            QMessageBox.warning(self, "AI отключен", "Настройте GEMINI_API_KEY, чтобы использовать чат.")
            return
        text = self.chat_input.toPlainText().strip()
        if not text:
            return
        self.chat_input.clear()
        self.chat_history.append({"role": "user", "content": text})
        self._append_chat_message("Вы", text)
        self.chat_send_button.setEnabled(False)
        self.statusBar().showMessage("ИИ пишет ответ…", 0)

        history_copy = [msg.copy() for msg in self.chat_history]
        self.chat_thread = ChatWorker(
            self.ai_client,
            self.editor.toPlainText(),
            history_copy,
        )
        self.chat_thread.completed.connect(self._chat_reply_ready)
        self.chat_thread.failed.connect(self._chat_failed)
        self.chat_thread.finished.connect(self._cleanup_chat_thread)
        self.chat_thread.start()

    def _chat_reply_ready(self, reply: str) -> None:
        self.chat_history.append({"role": "assistant", "content": reply})
        self._append_chat_message("Aurora AI", reply)
        self.statusBar().showMessage("Ответ получен", 3000)

    def _chat_failed(self, error: str) -> None:
        self.statusBar().showMessage("Ошибка чата", 3000)
        QMessageBox.critical(self, "Ошибка ИИ", error)
        if self.chat_history and self.chat_history[-1].get("role") == "user":
            self.chat_history.pop()

    def _cleanup_chat_thread(self) -> None:
        self.chat_thread = None
        self.chat_send_button.setEnabled(self.ai_client is not None)

    def _append_chat_message(self, author: str, text: str) -> None:
        if not text:
            return
        escaped = text.replace("\n", "<br>")
        self.chat_view.append(f"<b>{author}:</b><br>{escaped}\n")

    # endregion ---------------------------------------------------------


def ensure_store() -> NoteStore:
    data_dir = Path.home() / ".aurora_notes"
    return NoteStore(data_dir)


def ensure_settings_store() -> SettingsStore:
    data_dir = Path.home() / ".aurora_notes"
    return SettingsStore(data_dir / "settings.json")


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setWindowIcon(QIcon())

    store = ensure_store()
    settings_store = ensure_settings_store()
    settings = settings_store.load()

    window = MainWindow(store, settings_store, settings)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
