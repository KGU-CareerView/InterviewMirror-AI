import argparse
import time

import cv2
import grpc
import numpy as np

from app import config
from generated import interview_pb2
from generated import interview_pb2_grpc


SERVER_ADDRESS = "127.0.0.1:50051"
SESSION_ID = "preprocess-compare-session"
USER_ID = "preprocess-compare-user"

INPUT_SIZE = getattr(config, "INPUT_SIZE", 224)
MEAN = np.array(getattr(config, "MEAN", [0.485, 0.456, 0.406]), dtype=np.float32)
STD = np.array(getattr(config, "STD", [0.229, 0.224, 0.225]), dtype=np.float32)
BBOX_MARGIN = getattr(config, "BBOX_MARGIN", 0.25)


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


def crop_face(frame, bbox):
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]

    box_w = x2 - x1
    box_h = y2 - y1

    mx = int(box_w * BBOX_MARGIN)
    my = int(box_h * BBOX_MARGIN)

    nx1 = max(0, x1 - mx)
    ny1 = max(0, y1 - my)
    nx2 = min(w, x2 + mx)
    ny2 = min(h, y2 + my)

    if nx2 <= nx1 or ny2 <= ny1:
        return frame

    return frame[ny1:ny2, nx1:nx2]


def preprocess(frame, use_crop, use_normalize):
    face_detected, bbox = detect_face(frame)

    if use_crop and face_detected:
        model_frame = crop_face(frame, bbox)
    else:
        model_frame = frame

    resized = cv2.resize(model_frame, (INPUT_SIZE, INPUT_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    arr = rgb.astype(np.float32) / 255.0

    if use_normalize:
        arr = (arr - MEAN) / STD

    tensor = np.transpose(arr, (2, 0, 1))
    tensor = np.expand_dims(tensor, axis=0).astype(np.float32)

    return tensor, face_detected, bbox


def make_request(frame, use_crop, use_normalize):
    tensor, face_detected, bbox = preprocess(
        frame=frame,
        use_crop=use_crop,
        use_normalize=use_normalize,
    )

    x1, y1, x2, y2 = bbox

    print(
        "[PREPROCESS]",
        "crop=", use_crop,
        "normalize=", use_normalize,
        "face=", face_detected,
        "mean=", round(float(tensor.mean()), 4),
        "std=", round(float(tensor.std()), 4),
        "min=", round(float(tensor.min()), 4),
        "max=", round(float(tensor.max()), 4),
    )

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


def run_mode(stub, frame, use_crop, use_normalize):
    request = make_request(frame, use_crop, use_normalize)
    response = stub.AnalyzeFrame(request, timeout=5)

    return response.label, response.confidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=1.5)
    args = parser.parse_args()

    channel = grpc.insecure_channel(SERVER_ADDRESS)
    stub = interview_pb2_grpc.InterviewAIServiceStub(channel)

    health = stub.HealthCheck(interview_pb2.HealthRequest(), timeout=5)
    print("[HEALTH]", health.status)

    cap = cv2.VideoCapture(getattr(config, "CAMERA_INDEX", 0))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, getattr(config, "FRAME_WIDTH", 1280))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, getattr(config, "FRAME_HEIGHT", 720))

    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        return

    modes = [
        ("crop+normalize", True, True),
        ("crop only", True, False),
        ("normalize only", False, True),
        ("raw only", False, False),
    ]

    last_time = 0
    last_results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()

        if now - last_time >= args.interval:
            last_results = []

            for mode_name, use_crop, use_normalize in modes:
                try:
                    label, confidence = run_mode(
                        stub=stub,
                        frame=frame,
                        use_crop=use_crop,
                        use_normalize=use_normalize,
                    )
                    result = f"{mode_name}: {label} {confidence:.3f}"
                    print("[RESULT]", result)
                    last_results.append(result)
                except grpc.RpcError as exc:
                    result = f"{mode_name}: ERROR {exc.code()}"
                    print("[ERROR]", result)
                    last_results.append(result)

            print("-" * 80)
            last_time = now

        y = 40
        for line in last_results:
            cv2.putText(
                frame,
                line,
                (20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            y += 32

        cv2.putText(
            frame,
            "Press q to quit",
            (20, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Preprocess Compare", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
