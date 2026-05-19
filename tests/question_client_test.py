import grpc

from generated import interview_pb2
from generated import interview_pb2_grpc


def main():
    channel = grpc.insecure_channel("127.0.0.1:50051")
    stub = interview_pb2_grpc.InterviewAIServiceStub(channel)

    request = interview_pb2.InitialQuestionGenerateRequest(
        session_id="question-test-session",
        user_id="question-test-user",
        category="백엔드 개발자",
        interview_type="기술 면접",
        difficulty="normal",
        question_count=3,
        time_per_question=60,
        resume_text="Python, Java, FastAPI, gRPC 프로젝트 경험이 있습니다.",
        language="ko-KR",
    )

    try:
        response = stub.GenerateInitialQuestions(request, timeout=20)
        print("[QUESTIONS]")
        for question in response.questions:
            print("-" * 60)
            print("index:", question.index)
            print("question:", question.question)
            print("tooltip:", question.tooltip)
            print("category:", question.category)
            print("intent:", question.intent)
            print("answer_keywords:", list(question.answer_keywords))
    except grpc.RpcError as exc:
        print("[ERROR] GenerateInitialQuestions failed.")
        print(exc.code(), exc.details())


if __name__ == "__main__":
    main()
