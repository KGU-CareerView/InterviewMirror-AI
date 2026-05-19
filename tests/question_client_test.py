import grpc

from generated import interview_pb2
from generated import interview_pb2_grpc


def test_initial_questions(stub):
    response = stub.GenerateInitialQuestions(
        interview_pb2.InitialQuestionGenerateRequest(
            session_id="test-session-001",
            user_id="test-user-001",
            category="백엔드",
            interview_type="기술 면접",
            difficulty="normal",
            question_count=3,
            time_per_question=60,
            resume_text="""
Spring Boot와 MySQL을 활용해 게시판 프로젝트를 구현했습니다.
JWT 기반 로그인, 게시글 CRUD, 댓글 기능, 파일 업로드 기능을 맡았습니다.
gRPC 기반 AI 서버 연동 경험이 있습니다.
""",
            language="ko-KR",
        )
    )

    print("\n========== INITIAL QUESTIONS ==========")

    for item in response.questions:
        print("=" * 60)
        print(f"[{item.index}] {item.question}")
        print(f"카테고리: {item.category}")
        print(f"의도: {item.intent}")
        print(f"툴팁: {item.tooltip}")
        print(f"키워드: {list(item.answer_keywords)}")


def test_follow_up_question(stub):
    response = stub.GenerateFollowUpQuestion(
        interview_pb2.FollowUpQuestionGenerateRequest(
            session_id="test-session-001",
            user_id="test-user-001",
            previous_question="gRPC 기반 AI 서버 연동 경험에 대해 설명해 주세요.",
            answer="""
REST API보다 실시간 스트리밍에 유리하다고 판단해서 gRPC를 사용했습니다.
프론트에서 추출한 특징점 데이터를 서버로 보내고, 서버는 ONNX 모델로 추론한 뒤 결과를 반환하는 구조였습니다.
""",
            category="백엔드",
            interview_type="기술 면접",
            difficulty="normal",
            resume_text="""
Spring Boot와 MySQL을 활용해 게시판 프로젝트를 구현했습니다.
JWT 기반 로그인, 게시글 CRUD, 댓글 기능, 파일 업로드 기능을 맡았습니다.
gRPC 기반 AI 서버 연동 경험이 있습니다.
""",
            history=[
                interview_pb2.InterviewTurn(
                    index=1,
                    question="본인이 맡았던 백엔드 프로젝트 경험을 설명해 주세요.",
                    answer="게시판 프로젝트에서 로그인, CRUD, 파일 업로드 기능을 구현했습니다.",
                )
            ],
            language="ko-KR",
        )
    )

    item = response.question

    print("\n========== FOLLOW-UP QUESTION ==========")
    print("=" * 60)
    print(f"[{item.index}] {item.question}")
    print(f"카테고리: {item.category}")
    print(f"의도: {item.intent}")
    print(f"툴팁: {item.tooltip}")
    print(f"키워드: {list(item.answer_keywords)}")


def main():
    channel = grpc.insecure_channel("127.0.0.1:50051")
    stub = interview_pb2_grpc.InterviewAIServiceStub(channel)

    health = stub.HealthCheck(interview_pb2.HealthRequest())

    print("[HEALTH]", health.status)
    print("[MODEL]", health.model_path)
    print("[QUESTION CLIENT]", health.question_client_status)

    test_initial_questions(stub)
    test_follow_up_question(stub)


if __name__ == "__main__":
    main()