import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from pydantic import ValidationError

from app.questions.schemas import (
    FollowUpQuestionGenerateResult,
    QuestionGenerateResult,
)

load_dotenv()


class GeminiQuestionClient:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
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
        question_count = self._normalize_question_count(question_count)
        prompt = self._build_initial_prompt(
            category=category,
            interview_type=interview_type,
            difficulty=difficulty,
            question_count=question_count,
            time_per_question=time_per_question,
            resume_text=resume_text,
            language=language,
        )
        return self._generate_json(prompt, QuestionGenerateResult)

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
        prompt = self._build_follow_up_prompt(
            previous_question=previous_question,
            answer=answer,
            category=category,
            interview_type=interview_type,
            difficulty=difficulty,
            resume_text=resume_text,
            history=history,
            language=language,
        )
        return self._generate_json(prompt, FollowUpQuestionGenerateResult)

    def _generate_json(self, prompt: str, schema_model):
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": schema_model,
            },
        )

        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            return parsed

        try:
            return schema_model.model_validate_json(response.text)
        except ValidationError as exc:
            raise ValueError(f"Invalid Gemini response: {exc}\nraw={response.text}") from exc

    def _normalize_question_count(self, question_count: int) -> int:
        if question_count <= 0:
            return 3
        if question_count > 10:
            return 10
        return question_count

    def _build_initial_prompt(
        self,
        category: str,
        interview_type: str,
        difficulty: str,
        question_count: int,
        time_per_question: int,
        resume_text: str,
        language: str,
    ) -> str:
        category = category.strip() or "종합"
        interview_type = interview_type.strip() or "기술 면접"
        difficulty = difficulty.strip() or "normal"
        resume_text = resume_text.strip() or "제공된 이력서/자기소개서 내용 없음"
        language = language.strip() or "ko-KR"
        time_per_question = time_per_question or 60

        return f"""
너는 AI 모의면접 서비스의 면접 질문 생성기다.

면접 조건:
- 카테고리: {category}
- 면접 유형: {interview_type}
- 난이도: {difficulty}
- 질문 수: {question_count}
- 질문당 답변 시간: {time_per_question}초
- 응답 언어: {language}

이력서/자기소개서/포트폴리오 내용:
{resume_text}

생성 규칙:
1. 실제 면접에서 물어볼 법한 자연스러운 질문을 만든다.
2. resume_text가 있으면 개인화 질문을 우선 생성한다.
3. 질문은 서로 의미가 겹치지 않게 만든다.
4. tooltip은 사용자가 답변 방향을 잡을 수 있게 짧게 작성한다.
5. category는 인성, 직무, 기술, 프로젝트, 경험, 꼬리질문, 종합 중 하나에 가깝게 작성한다.
6. intent는 이 질문으로 무엇을 확인하려는지 설명한다.
7. answer_keywords는 3~6개 작성한다.
8. 반드시 JSON schema에 맞게 반환한다.
""".strip()

    def _build_follow_up_prompt(
        self,
        previous_question: str,
        answer: str,
        category: str,
        interview_type: str,
        difficulty: str,
        resume_text: str,
        history: list[dict[str, Any]],
        language: str,
    ) -> str:
        return f"""
너는 AI 모의면접 서비스의 꼬리질문 생성기다.

면접 조건:
- 카테고리: {category or '종합'}
- 면접 유형: {interview_type or '기술 면접'}
- 난이도: {difficulty or 'normal'}
- 응답 언어: {language or 'ko-KR'}

이력서/자기소개서/포트폴리오 내용:
{resume_text or '제공된 내용 없음'}

이전 면접 이력:
{history}

직전 질문:
{previous_question}

지원자 답변:
{answer}

생성 규칙:
1. 지원자 답변에서 더 검증할 만한 부분을 하나 골라 꼬리질문 1개를 만든다.
2. 너무 공격적이지 않게, 실제 면접관처럼 자연스럽게 질문한다.
3. tooltip은 답변 방향을 짧게 알려준다.
4. category는 가능하면 꼬리질문으로 작성한다.
5. answer_keywords는 3~6개 작성한다.
6. 반드시 JSON schema에 맞게 반환한다.
""".strip()
