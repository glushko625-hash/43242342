"""Simple local filesystem-based note storage."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class Note:
    identifier: str
    title: str
    content: str


class NoteStore:
    """Stores notes in a JSON index with individual text files."""

    INDEX_FILE = "index.json"

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.base_dir / self.INDEX_FILE
        if not self.index_path.exists():
            self._write_index({})

    def _read_index(self) -> Dict[str, Dict[str, str]]:
        if self.index_path.exists():
            with self.index_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        return {}

    def _write_index(self, data: Dict[str, Dict[str, str]]) -> None:
        with self.index_path.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

    def list_notes(self) -> List[Note]:
        index = self._read_index()
        notes: List[Note] = []
        for identifier, metadata in index.items():
            path = self.base_dir / metadata["filename"]
            content = ""
            if path.exists():
                content = path.read_text(encoding="utf-8")
            notes.append(
                Note(
                    identifier=identifier,
                    title=metadata.get("title", "Untitled"),
                    content=content,
                )
            )
        return notes

    def create_note(self, title: str) -> Note:
        index = self._read_index()
        identifier = str(max([int(k) for k in index.keys()] or [0]) + 1)
        filename = f"note_{identifier}.md"
        index[identifier] = {"title": title, "filename": filename}
        self._write_index(index)
        (self.base_dir / filename).write_text("", encoding="utf-8")
        return Note(identifier=identifier, title=title, content="")

    def save_note(self, note: Note) -> None:
        index = self._read_index()
        if note.identifier not in index:
            index[note.identifier] = {"title": note.title, "filename": f"note_{note.identifier}.md"}
        else:
            index[note.identifier]["title"] = note.title
        self._write_index(index)
        filename = index[note.identifier]["filename"]
        (self.base_dir / filename).write_text(note.content, encoding="utf-8")

    def rename(self, note: Note, new_title: str) -> Note:
        note.title = new_title
        self.save_note(note)
        return note


__all__ = ["Note", "NoteStore"]
