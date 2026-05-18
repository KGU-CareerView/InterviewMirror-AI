import time
import threading

import cv2
import grpc

import interview_pb2
import interview_pb2_grpc


def request_generator(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield interview_pb2.FrameRequest(
            session_id="test-session-001",
            user_id="test-user-001",
            image=buffer.tobytes(),
            timestamp=int(time.time() * 1000),
        )


def receive_responses(responses):
    for response in responses:
        print(
            f"label={response.label}, "
            f"confidence={response.confidence:.4f}, "
            f"feedback={response.feedback}"
        )


def main():
    channel = grpc.insecure_channel("127.0.0.1:50051")
    stub = interview_pb2_grpc.InterviewAIServiceStub(channel)

    # health check
    health = stub.HealthCheck(interview_pb2.HealthRequest())
    print("[HEALTH]", health.status, health.model_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam open failed")
        return

    print("[INFO] Streaming started (press q to quit)")

    # 🔥 streaming 시작
    responses = stub.AnalyzeFrameStream(request_generator(cap))

    # 응답 처리 스레드
    t = threading.Thread(target=receive_responses, args=(responses,))
    t.daemon = True
    t.start()

    # 화면 출력 루프
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("gRPC Client Streaming", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
