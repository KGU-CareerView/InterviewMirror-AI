import os

from dotenv import load_dotenv
from google import genai
from pydantic import ValidationError

from questions.schemas import QuestionGenerateResult


load_dotenv()


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

    def generate_questions_with_tooltips(
        self,
        job_role: str,
        company_name: str,
        resume_text: str,
        interview_type: str,
        question_count: int,
        difficulty: str,
        language: str,
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
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": QuestionGenerateResult,
            },
        )

        parsed = getattr(response, "parsed", None)

        if parsed is not None:
            return parsed

        try:
            return QuestionGenerateResult.model_validate_json(response.text)
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
    ) -> str:
        job_role = job_role.strip() or "지원 직무 미지정"
        company_name = company_name.strip() or "회사명 미지정"
        resume_text = resume_text.strip() or "제공된 이력서/자기소개서 내용 없음"
        interview_type = interview_type.strip() or "종합"
        difficulty = difficulty.strip() or "normal"
        language = language.strip() or "ko-KR"

        return f"""
너는 AI 모의면접 서비스의 면접 질문 생성기다.

사용자 정보:
- 지원 직무: {job_role}
- 회사명: {company_name}
- 면접 유형: {interview_type}
- 난이도: {difficulty}
- 응답 언어: {language}

이력서/자기소개서/포트폴리오 내용:
{resume_text}

생성해야 할 것:
- 면접 질문 {question_count}개
- 각 질문별 tooltip
- 각 질문의 category
- 각 질문의 intent
- 답변에 포함하면 좋은 answer_keywords

규칙:
1. 질문은 실제 면접에서 물어볼 법한 자연스러운 문장으로 작성한다.
2. tooltip은 프론트에서 말풍선/도움말로 보여줄 문장이므로 짧고 실용적으로 작성한다.
3. tooltip에는 답변 구조를 알려준다. 예: "상황-역할-행동-결과 순서로 답변하면 좋아요."
4. resume_text가 있으면 그 내용에 기반한 개인화 질문을 우선 생성한다.
5. resume_text가 부족하면 직무 기반 일반 질문을 생성한다.
6. 같은 의미의 질문을 반복하지 않는다.
7. category는 다음 중 하나에 가깝게 작성한다: 인성, 직무, 기술, 프로젝트, 경험, 꼬리질문, 종합.
8. intent는 "이 질문을 통해 무엇을 확인하려는지"를 설명한다.
9. answer_keywords는 3~6개 정도로 작성한다.
10. 반드시 JSON schema에 맞는 데이터만 반환한다.
""".strip()
EOFcat > question_client_test.py <<'EOF'
import grpc

import interview_pb2
import interview_pb2_grpc


def main():
    channel = grpc.insecure_channel("127.0.0.1:50051")
    stub = interview_pb2_grpc.InterviewAIServiceStub(channel)

    health = stub.HealthCheck(interview_pb2.HealthRequest())
    print("[HEALTH]", health.status)
    print("[MODEL]", health.model_path)
    print("[QUESTION CLIENT]", health.question_client_status)

    response = stub.GenerateQuestions(
        interview_pb2.QuestionGenerateRequest(
            session_id="test-session-001",
            user_id="test-user-001",
            job_role="백엔드 개발자",
            company_name="CareerView",
            resume_text="""
Spring Boot와 MySQL을 활용해 게시판 프로젝트를 구현했습니다.
JWT 기반 로그인, 게시글 CRUD, 댓글 기능, 파일 업로드 기능을 맡았습니다.
gRPC 기반 AI 서버 연동 경험이 있습니다.
""",
            interview_type="직무",
            question_count=3,
            difficulty="normal",
            language="ko-KR",
        )
    )

    for item in response.questions:
        print("=" * 60)
        print(f"[{item.index}] {item.question}")
        print(f"카테고리: {item.category}")
        print(f"의도: {item.intent}")
        print(f"툴팁: {item.tooltip}")
        print(f"키워드: {list(item.answer_keywords)}")


if __name__ == "__main__":
    main()
