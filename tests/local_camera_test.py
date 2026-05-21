import time
from pathlib import Path

import cv2
import grpc
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from app import config
from generated import interview_pb2
from generated import interview_pb2_grpc


SERVER_ADDR = "127.0.0.1:50051"
CAMERA_INDEX = getattr(config, "CAMERA_INDEX", 0)
INPUT_SIZE = getattr(config, "INPUT_SIZE", 224)

BASE_DIR = Path(__file__).resolve().parent.parent
FACE_DETECTOR_MODEL_PATH = Path(
    getattr(
        config,
        "MEDIAPIPE_FACE_TASK_MODEL",
        BASE_DIR / "models" / "face_detector.task",
    )
)

MEAN = np.array(getattr(config, "MEAN", [0.485, 0.456, 0.406]), dtype=np.float32)
STD = np.array(getattr(config, "STD", [0.229, 0.224, 0.225]), dtype=np.float32)


def create_face_detector():
    if not FACE_DETECTOR_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"MediaPipe face detector model not found: {FACE_DETECTOR_MODEL_PATH}"
        )

    base_options = python.BaseOptions(
        model_asset_path=str(FACE_DETECTOR_MODEL_PATH)
    )

    options = vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=getattr(config, "MIN_DETECTION_CONFIDENCE", 0.5),
    )

    return vision.FaceDetector.create_from_options(options)


def detect_face_bbox_with_mediapipe(detector, frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    detection_result = detector.detect(mp_image)

    if not detection_result.detections:
        return None

    frame_h, frame_w = frame_bgr.shape[:2]

    best_detection = max(
        detection_result.detections,
        key=lambda detection: (
            detection.bounding_box.width * detection.bounding_box.height
        ),
    )

    bbox = best_detection.bounding_box

    x = int(bbox.origin_x)
    y = int(bbox.origin_y)
    w = int(bbox.width)
    h = int(bbox.height)

    margin = getattr(config, "BBOX_MARGIN", 0.25)
    mx = int(w * margin)
    my = int(h * margin)

    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(frame_w, x + w + mx)
    y2 = min(frame_h, y + h + my)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def preprocess_face(face_bgr):
    resized = cv2.resize(face_bgr, (INPUT_SIZE, INPUT_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    x = rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD

    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0).astype(np.float32)

    return x


def make_request(frame, bbox, session_id="local-camera-session", user_id="local-user"):
    if bbox is None:
        return interview_pb2.FeatureRequest(
            session_id=session_id,
            user_id=user_id,
            tensor_shape=[1, 3, INPUT_SIZE, INPUT_SIZE],
            features=[],
            timestamp=int(time.time() * 1000),
            face_detected=False,
            bbox=interview_pb2.BoundingBox(x1=0, y1=0, x2=0, y2=0),
        )

    x1, y1, x2, y2 = bbox
    face = frame[y1:y2, x1:x2]

    if face.size == 0:
        return interview_pb2.FeatureRequest(
            session_id=session_id,
            user_id=user_id,
            tensor_shape=[1, 3, INPUT_SIZE, INPUT_SIZE],
            features=[],
            timestamp=int(time.time() * 1000),
            face_detected=False,
            bbox=interview_pb2.BoundingBox(x1=0, y1=0, x2=0, y2=0),
        )

    tensor = preprocess_face(face)

    print(
        "[INPUT]",
        "mean:", round(float(tensor.mean()), 4),
        "std:", round(float(tensor.std()), 4),
        "min:", round(float(tensor.min()), 4),
        "max:", round(float(tensor.max()), 4),
    )

    return interview_pb2.FeatureRequest(
        session_id=session_id,
        user_id=user_id,
        tensor_shape=list(tensor.shape),
        features=tensor.reshape(-1).tolist(),
        timestamp=int(time.time() * 1000),
        face_detected=True,
        bbox=interview_pb2.BoundingBox(
            x1=int(x1),
            y1=int(y1),
            x2=int(x2),
            y2=int(y2),
        ),
    )


def main():
    print("[INFO] Using MediaPipe FaceDetector")
    print("[INFO] Face detector model:", FACE_DETECTOR_MODEL_PATH)

    detector = create_face_detector()

    channel = grpc.insecure_channel(SERVER_ADDR)
    stub = interview_pb2_grpc.InterviewAIServiceStub(channel)

    try:
        health = stub.HealthCheck(interview_pb2.HealthRequest(), timeout=3)
        print("[HEALTH]", health.status)
        print("[MODEL]", health.model_path)
        print("[QUESTION CLIENT]", health.question_client_status)
    except grpc.RpcError as exc:
        print("[ERROR] gRPC server is not running.")
        print(exc.code(), exc.details())
        detector.close()
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {CAMERA_INDEX}")
        print("Mac 설정 > 개인정보 보호 및 보안 > 카메라에서 터미널/VSCode 권한 확인해봐.")
        detector.close()
        return

    print("[INFO] Camera opened.")
    print("[INFO] Press q to quit.")

    last_sent_time = 0.0
    send_interval = 0.3

    current_text = "waiting..."

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("[WARN] Failed to read frame.")
                break

            bbox = detect_face_bbox_with_mediapipe(detector, frame)

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                current_text = "no face"

            now = time.time()

            if now - last_sent_time >= send_interval:
                request = make_request(frame, bbox)

                try:
                    response = stub.AnalyzeFrame(request, timeout=5)
                    current_text = f"{response.label} / {response.confidence:.2f}"
                    print("[ANALYZE]", current_text, "|", response.feedback)
                except grpc.RpcError as exc:
                    current_text = f"gRPC error: {exc.code()}"
                    print("[ERROR]", exc.code(), exc.details())

                last_sent_time = now

            cv2.putText(
                frame,
                current_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                "MediaPipe FaceDetector / Press q to quit",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            cv2.imshow("InterviewMirror Local Camera Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()
        print("[INFO] Camera released.")
        print("[INFO] MediaPipe detector closed.")


if __name__ == "__main__":
    main()
