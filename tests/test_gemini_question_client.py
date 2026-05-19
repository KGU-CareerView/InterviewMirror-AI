from app.questions.gemini_question_client import GeminiQuestionClient


def main():
    client = GeminiQuestionClient()

    result = client.generate_initial_questions(
        category="백엔드 개발자",
        interview_type="기술 면접",
        difficulty="normal",
        question_count=3,
        time_per_question=60,
        resume_text="""
지원자는 Python, Java, FastAPI, gRPC를 사용한 프로젝트 경험이 있습니다.
AI 모의면접 시스템에서 gRPC 서버와 Gemini 기반 질문 생성 기능을 담당했습니다.
""",
        language="ko-KR",
    )

    print("\n[Gemini Initial Questions Test]")
    print("questions count:", len(result.questions))

    for q in result.questions:
        print("-" * 60)
        print("index:", q.index)
        print("question:", q.question)
        print("tooltip:", q.tooltip)
        print("category:", q.category)
        print("intent:", q.intent)
        print("answer_keywords:", q.answer_keywords)


if __name__ == "__main__":
    main()
