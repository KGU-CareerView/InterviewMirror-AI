# Gemini subtitle client for interview audio transcription.
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import ValidationError

from app.subtitles.schemas import SubtitleResult

load_dotenv()


class GeminiSubtitleClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

        self.client = genai.Client(api_key=self.api_key)

    def generate_subtitles(
        self,
        audio_path: Path,
        language_hint: str = "ko-KR",
    ) -> SubtitleResult:
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        uploaded_file = self.client.files.upload(file=str(audio_path))

        prompt = (
            "You are a speech-to-subtitle module for an AI mock interview service.\n"
            "Listen to the uploaded audio and return only valid JSON.\n"
            f"The spoken language hint is: {language_hint}\n\n"
            "Return the result using this exact JSON structure:\n"
            "{\n"
            f'  "language": "{language_hint}",\n'
            '  "summary": "short summary of the spoken answer",\n'
            '  "segments": [\n'
            "    {\n"
            '      "index": 1,\n'
            '      "start_ms": 0,\n'
            '      "end_ms": 3000,\n'
            '      "text": "transcribed subtitle text"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "1. Transcribe the spoken answer naturally.\n"
            "2. If the audio is Korean, write the subtitle text in Korean.\n"
            "3. Split the subtitle into sentence-level or meaning-level segments.\n"
            "4. start_ms and end_ms must be integers in milliseconds.\n"
            "5. summary should be one or two short sentences.\n"
            "6. Do not include markdown.\n"
            "7. Do not include code fences.\n"
            "8. Return JSON only."
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, uploaded_file],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )

        response_text = response.text.strip()

        try:
            data = json.loads(response_text)
            return SubtitleResult.model_validate(data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Gemini response is not valid JSON: {response_text}") from exc
        except ValidationError as exc:
            raise ValueError(f"Invalid Gemini subtitle response: {exc}") from exc
