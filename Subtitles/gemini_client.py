
import os
from pathlib import Path

from google import genai
from pydantic import ValidationError

from subtitles.schemas import SubtitleResult


class GeminiSubtitleClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

        self.client = genai.Client(api_key=self.api_key)

    def generate_subtitles(
        self,
        audio_path: str | Path,
        language_hint: str = "ko-KR",
    ) -> SubtitleResult:
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        uploaded_file = self.client.files.upload(file=str(audio_path))

        prompt = self._build_prompt(language_hint)

        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, uploaded_file],
            config={
                "response_format": {
                    "text": {
                        "mime_type": "application/json",
                        "schema": SubtitleResult.model_json_schema(),
                    }
                }
            },
        )

        try:
            return SubtitleResult.model_validate_json(response.text)
        except ValidationError as exc:
            raise ValueError(f"Invalid Gemini subtitle response: {exc}") from exc

    def _build_prompt(self, language_hint: str) -> str:
        return f"""
You are an automatic subtitle generator for a mock interview service.

Task:
- Transcribe the interview answer from the audio.
- Generate subtitle segments.
- Keep each subtitle short enough to display on screen.
- Use natural Korean spacing and punctuation when the speech is Korean.
- Remove filler sounds only when they hurt readability.
- Do not invent content that is not in the audio.
- Return only data matching the provided JSON schema.

Subtitle rules:
- language_hint: {language_hint}
- start_ms and end_ms must be integers.
- index must start from 1.
- Each segment should be roughly 1 to 5 seconds long.
- Each text should preferably be under 42 Korean characters.
""".strip()