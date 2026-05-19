import json
import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from pydantic import ValidationError

from questions.schemas import (
    FollowUpQuestionGenerateResult,
    InitialQuestionGenerateResult,
    QuestionItemModel,
)

load_dotenv()


class GeminiQuestionClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
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
    ) -> InitialQuestionGenerateResult:
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

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": InitialQuestionGenerateResult,
            },
        )

        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            return parsed

        text = self._extract_response_text(response)

        try:
            return InitialQuestionGenerateResult.model_validate_json(text)
        except ValidationError as exc:
            raise ValueError(f"Invalid Gemini initial question response: {exc}") from exc

    def generate_follow_up_question(
        self,
        previous_question: str,
        answer: str,
        category: str,
        interview_type: str,
        difficulty: str,
        resume_text: str,
        history: list[dict[str, Any]] | None,
        language: str,
    ) -> FollowUpQuestionGenerateResult:
        prompt = self._build_follow_up_prompt(
            previous_question=previous_question,
            answer=answer,
            category=category,
            interview_type=interview_type,
            difficulty=difficulty,
            resume_text=resume_text,
            history=history or [],
            language=language,
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": FollowUpQuestionGenerateResult,
            },
        )

        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            return parsed

        text = self._extract_response_text(response)

        try:
            return FollowUpQuestionGenerateResult.model_validate_json(text)
        except ValidationError as exc:
            raise ValueError(f"Invalid Gemini follow-up question response: {exc}") from exc

    def _normalize_question_count(self, question_count: int) -> int:
        if question_count <= 0:
            return 3
        if question_count > 10:
            return 10
        return question_count

    def _extract_response_text(self, response: Any) -> str:
        text = getattr(response, "text", None)
        if not text:
            raise ValueError("Gemini response text is empty.")

        text = text.strip()

        if text.startswith("```json"):
            text = text.removeprefix("```json").strip()
        if text.startswith("```"):
            text = text.removeprefix("```").strip()
        if text.endswith("```"):
            text = text.removesuffix("```").strip()

        return text

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
        interview_type = interview_type.strip() or "일반 면접"
        difficulty = difficulty.strip() or "normal"
        resume_text = resume_text.strip() or "제공된 이력서/자기소개서 내용 없음"
        language = language.strip() or "ko-KR"

        if time_per_question <= 0:
            time_per_question = 60

        return f"""
너는 AI 모의면접 서비스의 면접 질문 생성기다.

이번 작업은 "초기 질문 생성"이다.
아직 사용자의 답변은 없으므로, 아래에 제공된 필드와 이력서/자기소개서 내용을 기반으로 첫 면접 질문들을 만들어야 한다.

[입력 정보]
- 질문 카테고리: {category}
- 면접 유형: {interview_type}
- 난이도: {difficulty}
- 생성할 질문 수: {question_count}
- 질문당 답변 시간: {time_per_question}초
- 응답 언어: {language}

[이력서/자기소개서/포트폴리오 내용]
{resume_text}

[생성 규칙]
1. 사용자의 답변이 아직 없으므로, 답변 기반 꼬리 질문을 만들지 마라.
2. resume_text가 충분하면 그 내용을 바탕으로 개인화된 질문을 만들어라.
3. resume_text가 부족하면 category, interview_type, difficulty를 기준으로 일반 초기 질문을 만들어라.
4. 질문은 실제 면접관이 묻는 자연스러운 문장으로 작성해라.
5. tooltip은 프론트에서 말풍선으로 보여줄 짧은 도움말이다.
6. tooltip에는 답변 방향이나 답변 구조를 알려줘라.
7. category는 입력 category와 맞추되, 필요하면 인성, 직무, 기술, 프로젝트, 경험, 종합 중 하나로 구체화해라.
8. intent에는 이 질문으로 확인하려는 평가 포인트를 적어라.
9. answer_keywords는 답변에 포함하면 좋은 핵심 키워드 3~6개로 작성해라.
10. 반드시 JSON만 반환해라. 마크다운 코드블록은 쓰지 마라.

[반환 JSON 형식]
{{
  "questions": [
    {{
      "index": 1,
      "question": "면접 질문",
      "tooltip": "답변 도움말",
      "category": "질문 카테고리",
      "intent": "질문 의도",
      "answer_keywords": ["키워드1", "키워드2", "키워드3"]
    }}
  ]
}}
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
        previous_question = previous_question.strip()
        answer = answer.strip()
        category = category.strip() or "꼬리질문"
        interview_type = interview_type.strip() or "일반 면접"
        difficulty = difficulty.strip() or "normal"
        resume_text = resume_text.strip() or "제공된 이력서/자기소개서 내용 없음"
        language = language.strip() or "ko-KR"

        history_text = self._format_history(history)

        return f"""
너는 AI 모의면접 서비스의 면접관이다.

이번 작업은 "답변 기반 꼬리 질문 생성"이다.
초기 질문을 새로 만드는 것이 아니라, 반드시 사용자의 직전 답변을 분석해서 더 깊게 물어볼 수 있는 후속 질문 1개를 만들어야 한다.

[입력 정보]
- 질문 카테고리: {category}
- 면접 유형: {interview_type}
- 난이도: {difficulty}
- 응답 언어: {language}

[이력서/자기소개서/포트폴리오 내용]
{resume_text}

[이전 면접 흐름]
{history_text}

[직전 질문]
{previous_question}

[사용자 답변]
{answer}

[생성 규칙]
1. 반드시 사용자 답변 내용에 근거한 꼬리 질문을 만들어라.
2. 답변과 무관한 새로운 초기 질문을 만들지 마라.
3. 답변이 모호하면 구체적인 사례, 수치, 역할, 과정, 결과를 묻는 질문을 만들어라.
4. 답변이 기술 내용이면 구현 방식, 선택 이유, 한계, 개선점, 트러블슈팅을 깊게 물어봐라.
5. 답변이 경험 내용이면 본인의 역할, 갈등 해결, 의사결정 근거, 배운 점을 깊게 물어봐라.
6. 답변이 너무 짧거나 부실하면 답변을 구체화하도록 유도하는 질문을 만들어라.
7. tooltip은 이 꼬리 질문에 답할 때 어떤 방향으로 답하면 좋은지 짧게 알려줘라.
8. intent에는 이 꼬리 질문으로 확인하려는 평가 포인트를 적어라.
9. answer_keywords는 답변에 포함하면 좋은 핵심 키워드 3~6개로 작성해라.
10. 반드시 JSON만 반환해라. 마크다운 코드블록은 쓰지 마라.

[반환 JSON 형식]
{{
  "question": {{
    "index": 1,
    "question": "답변 기반 꼬리 질문",
    "tooltip": "답변 도움말",
    "category": "꼬리질문",
    "intent": "질문 의도",
    "answer_keywords": ["키워드1", "키워드2", "키워드3"]
  }}
}}
""".strip()

    def _format_history(self, history: list[dict[str, Any]]) -> str:
        if not history:
            return "이전 면접 흐름 없음"

        lines: list[str] = []

        for item in history:
            index = item.get("index", 0)
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()

            if not question and not answer:
                continue

            lines.append(
                f"{index}. 질문: {question}\n"
                f"   답변: {answer}"
            )

        return "\n".join(lines) if lines else "이전 면접 흐름 없음"


def question_item_to_dict(item: QuestionItemModel) -> dict[str, Any]:
    return json.loads(item.model_dump_json())