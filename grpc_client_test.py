import time

import cv2
import grpc

import interview_pb2
import interview_pb2_grpc


def main():
    channel = grpc.insecure_channel("127.0.0.1:50051")
    stub = interview_pb2_grpc.InterviewAIServiceStub(channel)

    health = stub.HealthCheck(interview_pb2.HealthRequest())
    print("[HEALTH]", health.status, health.model_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Webcam open failed")
        return

    print("[INFO] Press q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        request = interview_pb2.FrameRequest(
            session_id="test-session-001",
            user_id="test-user-001",
            image=buffer.tobytes(),
            timestamp=int(time.time() * 1000),
        )

        response = stub.AnalyzeFrame(request)
        print(
            f"label={response.label}, "
            f"confidence={response.confidence:.4f}, "
            f"feedback={response.feedback}"
        )

        if response.face_detected:
            x1, y1, x2, y2 = response.bbox.x1, response.bbox.y1, response.bbox.x2, response.bbox.y2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{response.label} {response.confidence:.2f}",
                (x1, max(30, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        cv2.imshow("gRPC Client Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
