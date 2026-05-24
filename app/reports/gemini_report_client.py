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
        self.model = model or os.getenv("GEMINI_REPORT_MODEL", "gemini-3.1-pro-preview")

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
        emotion_graph_json: str = "",
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
            emotion_graph_json=emotion_graph_json,
            language=language,
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=FinalInterviewReport,
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
        emotion_graph_json: str,
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
            "emotion_graph_json": emotion_graph_json,
            "language": language,
        }

        return f"""
You are a senior AI interview diagnostician for a mock interview product.

Generate a detailed and professional final interview report from the input data.
The report must diagnose the candidate across four axes:
1. Facial emotion and expression analysis:
   - Use emotion_graph_json for the full-session expression trend.
   - Use each question_results[].emotion_result_json for question-level signals.
2. Answer content and accuracy:
   - Use question, answer, category, interview_type, difficulty, and resume_text.
   - Evaluate relevance, specificity, logical structure, job fit, technical accuracy, and consistency.
3. Answer length and response time:
   - Use answer_length and response_time_seconds.
   - Diagnose whether the answer was too short, verbose, delayed, rushed, or well-paced.
4. Voice analysis:
   - Use voice_score as the precomputed reference score.
   - Use audio_summary, zcr_samples, pitch_stability, energy_stability, pause_ratio, and voice_feedback
     to explain speech stability, volume consistency, pauses, speed, filler tendency, and delivery confidence.

The output language must follow this language code: {language}.

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
  "final_advice": "final advice",
  "question_feedbacks": [
    {{
      "index": 1,
      "total_score": 0,
      "content_score": 0,
      "voice_score": 0,
      "expression_score": 0,
      "overall_feedback": "question-level overall feedback",
      "content_feedback": "answer content feedback",
      "voice_feedback": "voice feedback",
      "expression_feedback": "expression feedback"
    }}
  ]
}}

Scoring rules:
- overall_score must be 0 to 100.
- overall_score must synthesize content_score, voice_score, expression_score, and timing/length quality.
- content_score must reflect answer completeness, relevance, specificity, and consistency.
- voice_score must reflect pitch stability, energy stability, pause ratio, and speech flow.
- expression_score must reflect facial expression stability, confidence, and nervousness signals.
- Each question_feedbacks item must match one input question_results[].index.
- Question total_score must synthesize content, voice, expression, answer length, and response time.
- Do not invent facts that are not supported by the transcript, resume, or analysis metrics.
- If a transcript is short or missing, lower content confidence and explain the limitation.
- Keep feedback concrete: mention observable evidence such as answer length, response time,
  speech_ratio, wpm, pause_count, filler_word_count, ttr, emotion labels, or score changes when available.
- time_based_insights must mention specific time ranges when scores changed noticeably.
- Do not include markdown.
- Do not include code fences.
- Return JSON only.

Input data:
{json.dumps(payload, ensure_ascii=True, indent=2)}
""".strip()
