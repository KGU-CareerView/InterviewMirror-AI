import json
import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import ValidationError

from app.reports.schemas import FinalInterviewReport

load_dotenv()


class GeminiReportClient:
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

    def generate_final_report(
        self,
        session_id: str,
        user_id: str,
        category: str,
        interview_type: str,
        difficulty: str,
        resume_text: str,
        question_results: list[dict[str, Any]],
        timeline_scores: list[dict[str, Any]],
        language: str = "ko-KR",
    ) -> FinalInterviewReport:
        prompt = self._build_prompt(
            session_id=session_id,
            user_id=user_id,
            category=category,
            interview_type=interview_type,
            difficulty=difficulty,
            resume_text=resume_text,
            question_results=question_results,
            timeline_scores=timeline_scores,
            language=language,
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
            ),
        )

        response_text = response.text.strip()

        try:
            data = json.loads(response_text)
            return FinalInterviewReport.model_validate(data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Gemini report response is not valid JSON: {response_text}") from exc
        except ValidationError as exc:
            raise ValueError(f"Invalid Gemini report response: {exc}") from exc

    def _build_prompt(
        self,
        session_id: str,
        user_id: str,
        category: str,
        interview_type: str,
        difficulty: str,
        resume_text: str,
        question_results: list[dict[str, Any]],
        timeline_scores: list[dict[str, Any]],
        language: str,
    ) -> str:
        payload = {
            "session_id": session_id,
            "user_id": user_id,
            "category": category,
            "interview_type": interview_type,
            "difficulty": difficulty,
            "resume_text": resume_text,
            "question_results": question_results,
            "timeline_scores": timeline_scores,
            "language": language,
        }

        return f"""
You are an AI interview coach.

Generate a final interview report based on:
1. Interview questions
2. User answers transcribed from speech
3. Follow-up questions
4. Voice tone stability scores
5. Facial expression scores
6. Time-based behavior score timeline

The output language must follow this language code: {language}

Return ONLY valid JSON with this exact structure:

{{
  "overall_summary": "overall summary",
  "overall_score": 0,
  "content_score": 0,
  "voice_score": 0,
  "expression_score": 0,
  "strengths": [
    {{
      "title": "strength title",
      "detail": "detailed explanation"
    }}
  ],
  "weaknesses": [
    {{
      "title": "weakness title",
      "detail": "detailed explanation",
      "improvement": "specific improvement suggestion"
    }}
  ],
  "time_based_insights": [
    {{
      "time_range": "00:00-00:30",
      "observation": "observed pattern",
      "suggestion": "suggestion"
    }}
  ],
  "final_advice": "final advice"
}}

Scoring rules:
- overall_score must be 0 to 100.
- content_score must reflect answer completeness, relevance, specificity, and consistency.
- voice_score must reflect pitch stability, energy stability, pause ratio, and speech flow.
- expression_score must reflect facial expression stability, confidence, and nervousness signals.
- time_based_insights must mention specific time ranges when scores changed noticeably.
- Do not include markdown.
- Do not include code fences.
- Return JSON only.

Input data:
{json.dumps(payload, ensure_ascii=True, indent=2)}
""".strip()
