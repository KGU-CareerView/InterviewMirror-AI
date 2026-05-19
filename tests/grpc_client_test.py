import time

import grpc
import numpy as np

from generated import interview_pb2
from generated import interview_pb2_grpc


def make_dummy_request():
    shape = [1, 3, 224, 224]
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
    channel = grpc.insecure_channel("127.0.0.1:50051")
    stub = interview_pb2_grpc.InterviewAIServiceStub(channel)

    try:
        health = stub.HealthCheck(interview_pb2.HealthRequest(), timeout=3)
        print("[HEALTH]", health.status)
        print("[MODEL]", health.model_path)
        print("[QUESTION CLIENT]", health.question_client_status)
    except grpc.RpcError as exc:
        print("[ERROR] gRPC server is not running.")
        print(exc.code(), exc.details())
        return

    request = make_dummy_request()

    try:
        response = stub.AnalyzeFrame(request, timeout=5)
        print("[ANALYZE]")
        print("label:", response.label)
        print("confidence:", response.confidence)
        print("feedback:", response.feedback)
    except grpc.RpcError as exc:
        print("[ERROR] AnalyzeFrame failed.")
        print(exc.code(), exc.details())


if __name__ == "__main__":
    main()
