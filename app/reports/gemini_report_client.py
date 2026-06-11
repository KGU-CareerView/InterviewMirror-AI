import json
import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import errors
from google.genai import types
from pydantic import ValidationError

from app.reports.schemas import FinalInterviewReport

load_dotenv()

DEFAULT_REPORT_MODEL = "gemini-3.5-flash"
DEFAULT_REPORT_FALLBACK_MODEL = "gemini-2.5-flash"


class GeminiReportClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_REPORT_MODEL", DEFAULT_REPORT_MODEL)
        self.fallback_model = os.getenv(
            "GEMINI_REPORT_FALLBACK_MODEL",
            DEFAULT_REPORT_FALLBACK_MODEL,
        )
        self.timeout_ms = int(os.getenv("GEMINI_REPORT_TIMEOUT_MS", "120000"))

        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

        self.client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(timeout=self.timeout_ms),
        )

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

        response = self._generate_content_with_fallback(prompt)

        response_text = response.text.strip()

        try:
            data = json.loads(response_text)
            return FinalInterviewReport.model_validate(data)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Gemini report response is not valid JSON: {response_text}"
            ) from exc
        except ValidationError as exc:
            raise ValueError(f"Invalid Gemini report response: {exc}") from exc

    def _generate_content_with_fallback(self, prompt: str):
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=FinalInterviewReport,
        )

        try:
            return self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )
        except errors.APIError as exc:
            if (
                not self._is_resource_exhausted(exc)
                or self.model == self.fallback_model
            ):
                raise

            print(
                "[WARN] Gemini report model quota exhausted; "
                f"retrying with fallback model: {self.fallback_model}"
            )
            return self.client.models.generate_content(
                model=self.fallback_model,
                contents=prompt,
                config=config,
            )

    def _is_resource_exhausted(self, exc: errors.APIError) -> bool:
        return exc.code == 429 or exc.status == "RESOURCE_EXHAUSTED"

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
The report must diagnose the candidate across five distinct axes:

1. EXPRESSION ANALYSIS (30% weight - INDEPENDENT from content and voice):
   - Use ONLY emotion_graph_json and each question_results[].emotion_result_json for scoring.
   - IGNORE the pre-computed expression_score field - recalculate based ONLY on emotion labels.
   - Score stable_confident expressions significantly higher (80-100 range).
   - Score neutral expressions moderately (60-75 range).
   - Score nervous_anxious expressions lower (30-50 range).
   - Analyze consistency and confidence throughout the interview.
   - This score must be INDEPENDENT and NOT influenced by content_score or voice_score.

2. ANSWER CONTENT EVALUATION (40% weight):
   - Use question, answer, category, interview_type, difficulty, and resume_text.
   - Evaluate relevance, specificity, logical structure, job fit, technical accuracy, and consistency.
   - If transcript_status is "placeholder_no_transcript" or answer_is_placeholder is true,
     treat the answer as missing. Do not evaluate raw_answer as user content.
   - Provide separate content_feedback distinct from expression and voice feedback.

3. Answer length and response time:
   - Use answer_length and response_time_seconds.
   - Diagnose whether the answer was too short, verbose, delayed, rushed, or well-paced.
   - Factor into overall question score but keep separate from content quality.

4. VOICE ANALYSIS (30% weight):
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

IMPORTANT: All score fields (overall_score, content_score, voice_score, expression_score, total_score) must be integers (0-100 range).
Do not use decimal points. Round scores to the nearest integer.

Scoring rules:
- All scores (overall_score, content_score, voice_score, expression_score, total_score, question scores) must be INTEGERS from 0 to 100.
- Round all scores to the nearest integer. Do not use decimals or floating-point values.
- overall_score must synthesize content_score (40%), voice_score (30%), expression_score (30%).
- IMPORTANT: These three scores must be calculated INDEPENDENTLY:
  * content_score: ONLY from answer text, question relevance, and resume fit. Ignore expression and voice.
  * voice_score: ONLY from voice metrics (pitch_stability, energy_stability, pause_ratio, etc). Ignore content and expression.
  * expression_score: ONLY from emotion_result_json and emotion_graph_json labels. Ignore content and voice.

- content_score must reflect answer completeness, relevance, specificity, and consistency. Only evaluate actual transcribed answers.

- voice_score must reflect pitch stability, energy stability, pause ratio, and speech flow.

- expression_score must ONLY be based on emotion labels from emotion_result_json:
  * stable_confident label → 80-100 (highest confidence and stability)
  * neutral label → 60-75 (acceptable but less confident)
  * nervous_anxious label → 30-50 (shows nervousness and lacks confidence)
  * Compute question expression_score as average of all emotion labels in that question, round to integer.
  * Compute session expression_score as average across all questions, round to integer.

- Each question_feedbacks item must match one input question_results[].index.
- Question total_score = content_score (40%) + voice_score (30%) + expression_score (30%).

- Do not invent facts that are not supported by the transcript, resume, or analysis metrics.
- If a transcript is short or missing, lower content_score and explain the limitation in content_feedback.
- Keep feedback concrete: mention observable evidence such as answer length, response time,
  speech_ratio, wpm, pause_count, filler_word_count, ttr, emotion labels, or score changes when available.
- time_based_insights must mention specific time ranges when scores changed noticeably.
- Do not include markdown.
- Do not include code fences.
- Return JSON only.

Input data:
{json.dumps(payload, ensure_ascii=True, indent=2)}
""".strip()
