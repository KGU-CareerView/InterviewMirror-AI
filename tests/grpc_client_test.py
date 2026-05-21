import time

import grpc
import numpy as np

from generated import interview_pb2
from generated import interview_pb2_grpc


SERVER_ADDRESS = "127.0.0.1:50051"


def make_dummy_request():
    shape = [1, 3, 224, 224]

    # 모델 입력 형태에 맞춘 dummy tensor
    features = np.zeros(shape, dtype=np.float32).reshape(-1).tolist()

    return interview_pb2.FeatureRequest(
        session_id="test-session-001",
        user_id="test-user-001",
        tensor_shape=shape,
        features=features,
        timestamp=int(time.time() * 1000),
        face_detected=True,
        bbox=interview_pb2.BoundingBox(
            x1=0,
            y1=0,
            x2=224,
            y2=224,
        ),
    )


def main():
    channel = grpc.insecure_channel(SERVER_ADDRESS)
    stub = interview_pb2_grpc.InterviewAIServiceStub(channel)

    print("[INFO] Checking gRPC server health...")

    health = stub.HealthCheck(interview_pb2.HealthRequest(), timeout=5)

    print("[HEALTH]", health.status)
    print("[MODEL]", health.model_path)
    print("[QUESTION CLIENT]", health.question_client_status)
    print("[SUBTITLE CLIENT]", health.subtitle_client_status)
    print("[VOICE TONE ANALYZER]", health.voice_tone_analyzer_status)

    print("\n[INFO] Sending dummy frame request...")

    request = make_dummy_request()
    response = stub.AnalyzeFrame(request, timeout=5)

    print("\n[ANALYSIS RESPONSE]")
    print("session_id:", response.session_id)
    print("user_id:", response.user_id)
    print("label:", response.label)
    print("confidence:", response.confidence)
    print("feedback:", response.feedback)
    print("timestamp:", response.timestamp)
    print("face_detected:", response.face_detected)
    print(
        "bbox:",
        response.bbox.x1,
        response.bbox.y1,
        response.bbox.x2,
        response.bbox.y2,
    )


if __name__ == "__main__":
    main()
