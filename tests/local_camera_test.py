import time

import cv2
import grpc
import numpy as np

from app import config
from generated import interview_pb2
from generated import interview_pb2_grpc
from app.face_cropper import FaceCropper


INPUT_SIZE = getattr(config, "INPUT_SIZE", 224)
CAMERA_INDEX = getattr(config, "CAMERA_INDEX", 0)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    tensor = rgb.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    tensor = (tensor - mean) / std

    tensor = np.transpose(tensor, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0)

    return tensor.astype(np.float32)


def main():
    channel = grpc.insecure_channel("127.0.0.1:50051")
    stub = interview_pb2_grpc.InterviewAIServiceStub(channel)

    try:
        health = stub.HealthCheck(interview_pb2.HealthRequest())
        print("[HEALTH]", health.status)
        print("[MODEL]", health.model_path)

        if hasattr(health, "question_client_status"):
            print("[QUESTION CLIENT]", health.question_client_status)

    except grpc.RpcError as exc:
        print("[ERROR] gRPC server is not running.")
        print(exc.code(), exc.details())
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("[ERROR] Webcam open failed.")
        return

    face_cropper = FaceCropper(
        model_path=getattr(config, "MEDIAPIPE_FACE_TASK_MODEL", "face_detector.task"),
        min_detection_confidence=getattr(config, "MIN_DETECTION_CONFIDENCE", 0.6),
        bbox_margin=getattr(config, "BBOX_MARGIN", 0.25),
    )

    print("[INFO] Local camera test started.")
    print("[INFO] Press q to quit.")

    last_result_text = "waiting..."

while True:
    ret, frame = cap.read()

    if not ret:
        print("[ERROR] Failed to read webcam frame.")
        break

    tensor = preprocess_frame(frame)

    print(
    "[INPUT DEBUG]",
    "shape=", tensor.shape,
    "min=", float(tensor.min()),
    "max=", float(tensor.max()),
    "mean=", float(tensor.mean()),
    "std=", float(tensor.std()),
)

    request = interview_pb2.FeatureRequest(
        session_id="local-camera-test",
        user_id="local-user",
        tensor_shape=list(tensor.shape),
        features=tensor.flatten().tolist(),
        timestamp=int(time.time() * 1000),
        face_detected=True,
        bbox=interview_pb2.BoundingBox(
            x1=0,
            y1=0,
            x2=frame.shape[1],
            y2=frame.shape[0],
        ),
    )

    try:
        response = stub.AnalyzeFrame(request)

        result_text = f"{response.label} / {response.confidence:.2f}"

        print(
            f"label={response.label}, "
            f"confidence={response.confidence:.4f}, "
            f"feedback={response.feedback}"
        )

    except grpc.RpcError as exc:
        result_text = f"gRPC error: {exc.code().name}"
        print("[gRPC ERROR]", exc.code(), exc.details())

    cv2.putText(
        frame,
        result_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    cv2.imshow("InterviewMirror Local Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    face_cropper.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()