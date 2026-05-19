import time

import cv2
import grpc
import numpy as np

from generated import interview_pb2
from generated import interview_pb2_grpc


SERVER_ADDR = "127.0.0.1:50051"
CAMERA_INDEX = 0
INPUT_SIZE = 224

# ImageNet normalize
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def detect_face_bbox(frame):
    """
    OpenCV Haar Cascade로 얼굴 bbox 찾기.
    MediaPipe 없이도 로컬 테스트 가능하게 만든 버전.
    return: (x1, y1, x2, y2) or None
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )

    if len(faces) == 0:
        return None

    # 가장 큰 얼굴 선택
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

    # margin 추가
    margin = 0.25
    mx = int(w * margin)
    my = int(h * margin)

    x1 = max(0, x - mx)
    y1 = max(0, y - my)
    x2 = min(frame.shape[1], x + w + mx)
    y2 = min(frame.shape[0], y + h + my)

    return x1, y1, x2, y2


def preprocess_face(face_bgr):
    """
    BGR face image -> NCHW float32 tensor
    """
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

    tensor = preprocess_face(face)

    # 입력이 진짜 바뀌는지 확인용
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
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {CAMERA_INDEX}")
        print("Mac 설정 > 개인정보 보호 및 보안 > 카메라 권한 확인해봐.")
        return

    print("[INFO] Camera opened.")
    print("[INFO] Press q to quit.")

    last_sent_time = 0
    send_interval = 0.8

    current_text = "waiting..."

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("[WARN] Failed to read frame.")
                break

            bbox = detect_face_bbox(frame)

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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
                "Press q to quit",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow("InterviewMirror Local Camera Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released.")


if __name__ == "__main__":
    main()
