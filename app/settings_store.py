"""Application settings persistence for Aurora Notes."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any


@dataclass
class AppSettings:
    """Serializable configuration for AI integration."""

    api_key: str = ""
    model: str = "gemini-2.5-pro"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppSettings":
        return cls(
            api_key=str(data.get("api_key", "")),
            model=str(data.get("model", "gemini-2.5-pro")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"api_key": self.api_key, "model": self.model}


@dataclass
class SettingsStore:
    """Reads and writes :class:`AppSettings` to a JSON file."""

    path: Path
    _defaults: AppSettings = field(default_factory=AppSettings)

    def load(self) -> AppSettings:
        if not self.path.exists():
            return AppSettings.from_dict(self._defaults.to_dict())
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return AppSettings.from_dict(self._defaults.to_dict())
        return AppSettings.from_dict(data)

    def save(self, settings: AppSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(settings.to_dict(), fh, ensure_ascii=False, indent=2)


__all__ = ["AppSettings", "SettingsStore"]
