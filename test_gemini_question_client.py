from dotenv import load_dotenv

from gemini_question_client import GeminiQuestionClient


def main():
    load_dotenv()

    client = GeminiQuestionClient()

    print("=== 초기 질문 생성 테스트 ===")
    initial_questions = client.generate_initial_questions(
        job_position="백엔드 개발자",
        company_name="InterviewMirror",
        interview_type="기술 면접",
        difficulty="normal",
        resume_text="Spring Boot와 gRPC 기반 프로젝트 경험이 있습니다.",
        portfolio_text="AI 면접 분석 시스템에서 서버 연동을 담당했습니다.",
        keywords=["gRPC", "Spring Boot", "AI 면접", "Docker"],
        question_count=3,
    )

    for q in initial_questions:
        print(q)

    print("\n=== 답변 기반 꼬리 질문 생성 테스트 ===")
    follow_up_questions = client.generate_follow_up_questions(
        job_position="백엔드 개발자",
        company_name="InterviewMirror",
        interview_type="기술 면접",
        difficulty="normal",
        previous_question="gRPC를 사용한 이유가 무엇인가요?",
        user_answer="실시간으로 면접 분석 데이터를 주고받아야 해서 REST보다 스트리밍에 유리한 gRPC를 사용했습니다.",
        answer_summary="지원자는 실시간 스트리밍 처리 때문에 gRPC를 선택했다고 답변함.",
        keywords=["gRPC", "streaming", "REST"],
        question_count=2,
    )

    for q in follow_up_questions:
        print(q)


if __name__ == "__main__":
    main()