import time

import cv2
import grpc
import numpy as np

from generated import interview_pb2
from generated import interview_pb2_grpc


SERVER_ADDRESS = "127.0.0.1:50051"
SESSION_ID = "local-camera-session"
USER_ID = "local-camera-user"

INPUT_SIZE = 224
SEND_INTERVAL_SEC = 0.5


def preprocess_frame(frame):
    resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    arr = rgb.astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)

    return arr


def detect_face(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        return False, (0, 0, 0, 0)

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

    return True, (x, y, x + w, y + h)


def make_request(frame):
    tensor = preprocess_frame(frame)
    face_detected, bbox = detect_face(frame)

    x1, y1, x2, y2 = bbox

    return interview_pb2.FeatureRequest(
        session_id=SESSION_ID,
        user_id=USER_ID,
        tensor_shape=list(tensor.shape),
        features=tensor.reshape(-1).tolist(),
        timestamp=int(time.time() * 1000),
        face_detected=face_detected,
        bbox=interview_pb2.BoundingBox(
            x1=int(x1),
            y1=int(y1),
            x2=int(x2),
            y2=int(y2),
        ),
    )


def draw_result(frame, response):
    label = response.label
    confidence = response.confidence
    feedback = response.feedback

    if response.face_detected:
        bbox = response.bbox
        cv2.rectangle(
            frame,
            (bbox.x1, bbox.y1),
            (bbox.x2, bbox.y2),
            (0, 255, 0),
            2,
        )
    else:
        cv2.putText(
            frame,
            "NO FACE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2,
        )

    cv2.putText(
        frame,
        f"Label: {label}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence:.3f}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )

    print("=" * 80)
    print("[MODEL RESULT]")
    print("label:", label)
    print("confidence:", confidence)
    print("feedback:", feedback)
    print("=" * 80)

    return frame


def main():
    channel = grpc.insecure_channel(SERVER_ADDRESS)
    stub = interview_pb2_grpc.InterviewAIServiceStub(channel)

    try:
        health = stub.HealthCheck(interview_pb2.HealthRequest(), timeout=5)
        print("[HEALTH]", health.status)
        print("[MODEL]", health.model_path)
        print("[QUESTION CLIENT]", health.question_client_status)
        print("[SUBTITLE CLIENT]", health.subtitle_client_status)
        print("[VOICE TONE ANALYZER]", health.voice_tone_analyzer_status)
    except grpc.RpcError as exc:
        print("[ERROR] gRPC server is not available.")
        print(exc.code(), exc.details())
        print()
        print("Run this first:")
        print("PYTHONPATH=. python -m app.grpc_server")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        print("On macOS, allow camera permission for Terminal or VS Code.")
        return

    print("[INFO] Local camera started.")
    print("[INFO] Press 'q' to quit.")

    last_sent_time = 0
    last_response = None

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        now = time.time()

        if now - last_sent_time >= SEND_INTERVAL_SEC:
            request = make_request(frame)

            try:
                last_response = stub.AnalyzeFrame(request, timeout=5)
            except grpc.RpcError as exc:
                print("[ERROR] AnalyzeFrame failed.")
                print(exc.code(), exc.details())
                last_response = None

            last_sent_time = now

        if last_response is not None:
            frame = draw_result(frame, last_response)

        cv2.imshow("InterviewMirror AI - Local Camera Test", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
