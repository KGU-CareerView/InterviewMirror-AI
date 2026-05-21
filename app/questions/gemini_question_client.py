import json
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import ValidationError

from app.questions.schemas import QuestionGenerateResult
from app.questions.schemas import QuestionItemModel

load_dotenv()


@dataclass
class FollowUpQuestionGenerateResult:
    question: QuestionItemModel


class GeminiQuestionClient:
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

    def generate_initial_questions(
        self,
        category: str,
        interview_type: str,
        difficulty: str,
        question_count: int,
        time_per_question: int,
        resume_text: str,
        language: str,
    ) -> QuestionGenerateResult:
        return self.generate_questions_with_tooltips(
            job_role=category or "unspecified job role",
            company_name="unspecified company",
            resume_text=resume_text or "",
            interview_type=interview_type or "general",
            question_count=question_count or 3,
            difficulty=difficulty or "normal",
            language=language or "ko-KR",
            time_per_question=time_per_question or 60,
            follow_up_context=None,
        )

    def generate_follow_up_question(
        self,
        previous_question: str,
        answer: str,
        category: str,
        interview_type: str,
        difficulty: str,
        resume_text: str,
        history: list[dict[str, Any]],
        language: str,
    ) -> FollowUpQuestionGenerateResult:
        follow_up_context = {
            "previous_question": previous_question or "",
            "user_answer": answer or "",
            "history": history or [],
            "resume_text": resume_text or "",
        }

        result = self.generate_questions_with_tooltips(
            job_role=category or "unspecified job role",
            company_name="unspecified company",
            resume_text=resume_text or "",
            interview_type=f"{interview_type or 'general'} follow-up",
            question_count=1,
            difficulty=difficulty or "normal",
            language=language or "ko-KR",
            time_per_question=60,
            follow_up_context=follow_up_context,
        )

        if not result.questions:
            fallback_question = QuestionItemModel(
                index=1,
                question="Could you explain your previous answer in more detail?",
                tooltip="Add a concrete example and explain your role clearly.",
                category="follow-up",
                intent="Evaluate answer depth and consistency.",
                answer_keywords=["example", "role", "result"],
            )
            return FollowUpQuestionGenerateResult(question=fallback_question)

        return FollowUpQuestionGenerateResult(question=result.questions[0])

    def generate_questions_with_tooltips(
        self,
        job_role: str,
        company_name: str,
        resume_text: str,
        interview_type: str,
        question_count: int,
        difficulty: str,
        language: str,
        time_per_question: int = 60,
        follow_up_context: dict[str, Any] | None = None,
    ) -> QuestionGenerateResult:
        question_count = self._normalize_question_count(question_count)

        prompt = self._build_prompt(
            job_role=job_role,
            company_name=company_name,
            resume_text=resume_text,
            interview_type=interview_type,
            question_count=question_count,
            difficulty=difficulty,
            language=language,
            time_per_question=time_per_question,
            follow_up_context=follow_up_context,
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
            return QuestionGenerateResult.model_validate(data)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Gemini question response is not valid JSON: {response_text}") from exc
        except ValidationError as exc:
            raise ValueError(f"Invalid Gemini question response: {exc}") from exc

    def _normalize_question_count(self, question_count: int) -> int:
        if question_count <= 0:
            return 3

        if question_count > 10:
            return 10

        return question_count

    def _build_prompt(
        self,
        job_role: str,
        company_name: str,
        resume_text: str,
        interview_type: str,
        question_count: int,
        difficulty: str,
        language: str,
        time_per_question: int,
        follow_up_context: dict[str, Any] | None,
    ) -> str:
        payload = {
            "job_role": job_role or "unspecified job role",
            "company_name": company_name or "unspecified company",
            "resume_text": resume_text or "",
            "interview_type": interview_type or "general",
            "question_count": question_count,
            "difficulty": difficulty or "normal",
            "language": language or "ko-KR",
            "time_per_question": time_per_question,
            "follow_up_context": follow_up_context,
        }

        safe_payload = json.dumps(payload, ensure_ascii=True, indent=2)

        return (
            "You are an interview question generation module for an AI mock interview service.\n"
            "Generate interview questions based on the given input data.\n\n"
            "The output language code is provided in the input JSON.\n"
            "If the language code is ko-KR, write the question text in Korean.\n\n"
            "Return ONLY valid JSON using this exact structure:\n"
            "{\n"
            '  "questions": [\n'
            "    {\n"
            '      "index": 1,\n'
            '      "question": "interview question",\n'
            '      "tooltip": "short practical answering tip",\n'
            '      "category": "question category",\n'
            '      "intent": "what this question evaluates",\n'
            '      "answer_keywords": ["keyword1", "keyword2", "keyword3"]\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "1. Generate realistic interview questions.\n"
            "2. Generate exactly question_count questions unless it is a follow-up request.\n"
            "3. If follow_up_context exists, generate one follow-up question based on the previous question and user answer.\n"
            "4. If resume_text exists, personalize the questions using it.\n"
            "5. Avoid duplicate questions.\n"
            "6. tooltip must be short and practical.\n"
            "7. category should be one of: personality, job, technical, project, experience, follow-up, general.\n"
            "8. intent must explain what the interviewer wants to evaluate.\n"
            "9. answer_keywords must include 3 to 6 useful points.\n"
            "10. Do not include markdown.\n"
            "11. Do not include code fences.\n"
            "12. Return JSON only.\n\n"
            "Input JSON:\n"
            f"{safe_payload}"
        )
