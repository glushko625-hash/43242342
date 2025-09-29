"""AI integration for contextual text editing and chatting using Google Gemini."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from google import genai
from google.genai import types


class GeminiError(RuntimeError):
    """Raised when the Gemini API call fails."""


@dataclass
class GeminiConfig:
    """Configuration for the Gemini client."""

    api_key: Optional[str]
    model: str = "gemini-2.5-pro"

    @classmethod
    def from_env(cls) -> "GeminiConfig":
        return cls(api_key=os.environ.get("GEMINI_API_KEY"))


@dataclass(frozen=True)
class ModelSpec:
    """Describes a supported model and its optional capabilities."""

    model_id: str
    label: str
    description: str
    supports_thinking: bool = True
    supports_google_search: bool = True


_MODEL_SPECS: List[ModelSpec] = [
    ModelSpec(
        model_id="gemini-2.5-pro",
        label="Gemini 2.5 Pro",
        description="Флагманская модель с расширенными инструментами и режимом размышлений.",
        supports_thinking=True,
        supports_google_search=True,
    ),
    ModelSpec(
        model_id="gemma-3-27b-it",
        label="Gemma 3 27B IT",
        description="Легкая и быстрая модель без расширенного размышления и поиска.",
        supports_thinking=False,
        supports_google_search=False,
    ),
]

_MODEL_SPEC_MAP: Dict[str, ModelSpec] = {spec.model_id: spec for spec in _MODEL_SPECS}


def available_model_specs() -> List[ModelSpec]:
    """Return the list of known model presets in display order."""

    return list(_MODEL_SPECS)


def _resolve_model_spec(model_id: str) -> ModelSpec:
    """Return metadata for the requested model, falling back to a basic profile."""

    spec = _MODEL_SPEC_MAP.get(model_id)
    if spec:
        return spec
    return ModelSpec(
        model_id=model_id,
        label=model_id,
        description="Пользовательская модель без дополнительных возможностей.",
        supports_thinking=False,
        supports_google_search=False,
    )


class GeminiClient:
    """Small wrapper around the google-genai client used by the app."""

    def __init__(self, config: Optional[GeminiConfig] = None) -> None:
        self.config = config or GeminiConfig.from_env()
        if not self.config.api_key:
            raise GeminiError(
                "GEMINI_API_KEY is not configured. Please set it before using AI features."
            )
        self._client = genai.Client(api_key=self.config.api_key)
        self._spec = _resolve_model_spec(self.config.model)

    def _run_stream(self, contents: List[types.Content]) -> str:
        config_kwargs: Dict[str, object] = {}

        if self._spec.supports_google_search:
            config_kwargs["tools"] = [types.Tool(googleSearch=types.GoogleSearch())]

        if self._spec.supports_thinking:
            # google-genai evolves quickly; prefer the unlimited thinking budget when supported.
            for field in ("thinking_budget", "budget_tokens"):
                try:
                    config_kwargs["thinking_config"] = types.ThinkingConfig(**{field: -1})
                    break
                except Exception:  # pragma: no cover - depends on installed SDK version
                    continue

        config = types.GenerateContentConfig(**config_kwargs)

        try:
            chunks = self._client.models.generate_content_stream(
                model=self.config.model,
                contents=contents,
                config=config,
            )
        except Exception as exc:  # pragma: no cover - low-level SDK errors
            raise GeminiError(str(exc)) from exc

        parts: List[str] = []
        try:
            for chunk in chunks:
                if chunk.text:
                    parts.append(chunk.text)
        except Exception as exc:  # pragma: no cover - stream iteration errors
            raise GeminiError(str(exc)) from exc

        if not parts:
            raise GeminiError("The AI response was empty. Please try again.")

        return "".join(parts).strip()

    def transform_selection(
        self,
        note_content: str,
        selected_text: str,
        instruction: str,
    ) -> str:
        """Use Gemini to transform the selected text with full-note context."""

        if not selected_text.strip():
            raise GeminiError("Please select some text before requesting an AI edit.")
        if not instruction.strip():
            raise GeminiError("Instruction cannot be empty.")

        prompt = (
            "You are assisting with editing a note. \n"
            "The full note content is provided between <note> tags. \n"
            "The text between <selection> tags is currently selected by the user. \n"
            f"Instruction: {instruction}\n"
            "Return only the transformed selection with no additional commentary."
            "\n<note>\n"
            f"{note_content}\n"
            "</note>\n<selection>\n"
            f"{selected_text}\n"
            "</selection>"
        )

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        ]

        return self._run_stream(contents)

    def chat(self, note_content: str, history: List[Dict[str, str]]) -> str:
        """Have a contextual conversation about the current note."""

        if not history or history[-1]["role"] != "user":
            raise GeminiError("The conversation must end with a user message.")

        user_message = history[-1]["content"].strip()
        if not user_message:
            raise GeminiError("Message cannot be empty.")

        transcript_lines: List[str] = []
        for message in history[:-1]:
            role = message.get("role", "user")
            label = "Пользователь" if role == "user" else "ИИ"
            transcript_lines.append(f"{label}: {message.get('content', '')}")

        transcript = "\n".join(transcript_lines) if transcript_lines else "(нет истории)"
        prompt = (
            "Ты помощник, который помогает редактировать и анализировать заметки.\n"
            "Используй контекст заметки и историю переписки, чтобы отвечать полезно.\n"
            "Контекст заметки расположен внутри тегов <note>.\n"
            "История чата расположена внутри тегов <history>.\n"
            "Ответь развернуто, при необходимости предложи идеи по улучшению.\n"
            "<note>\n"
            f"{note_content}\n"
            "</note>\n"
            "<history>\n"
            f"{transcript}\n"
            "</history>\n"
            "<message>\n"
            f"{user_message}\n"
            "</message>"
        )

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        ]

        return self._run_stream(contents)


    @property
    def model_spec(self) -> ModelSpec:
        """Expose the resolved model specification for UI consumers."""

        return self._spec


__all__ = [
    "GeminiClient",
    "GeminiConfig",
    "GeminiError",
    "ModelSpec",
    "available_model_specs",
]
