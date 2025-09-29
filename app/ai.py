"""AI integration for contextual text editing using Google Gemini."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

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


class GeminiClient:
    """Small wrapper around the google-genai client used by the app."""

    def __init__(self, config: Optional[GeminiConfig] = None) -> None:
        self.config = config or GeminiConfig.from_env()
        if not self.config.api_key:
            raise GeminiError(
                "GEMINI_API_KEY is not configured. Please set it before using AI features."
            )
        self._client = genai.Client(api_key=self.config.api_key)

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

        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=(
                            "You are assisting with editing a note. \n"
                            "The full note content is provided between <note> tags. \n"
                            "The text between <selection> tags is currently selected by the user. \n"
                            "Instruction: "
                            + instruction
                            + "\n"
                            "Return only the transformed selection with no additional commentary."
                            "\n<note>\n"
                            + note_content
                            + "\n</note>\n<selection>\n"
                            + selected_text
                            + "\n</selection>"
                        )
                    )
                ],
            ),
        ]

        tools = [types.Tool(googleSearch=types.GoogleSearch())]
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=-1),
            tools=tools,
        )

        try:
            chunks = self._client.models.generate_content_stream(
                model=self.config.model,
                contents=contents,
                config=config,
            )
        except Exception as exc:  # pragma: no cover - low-level SDK errors
            raise GeminiError(str(exc)) from exc

        parts = []
        try:
            for chunk in chunks:
                if chunk.text:
                    parts.append(chunk.text)
        except Exception as exc:  # pragma: no cover - stream iteration errors
            raise GeminiError(str(exc)) from exc

        if not parts:
            raise GeminiError("The AI response was empty. Please try again.")

        return "".join(parts).strip()


__all__ = ["GeminiClient", "GeminiConfig", "GeminiError"]
