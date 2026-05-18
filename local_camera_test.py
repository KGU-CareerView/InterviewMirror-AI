import time

import cv2
import grpc
import numpy as np

import interview_pb2
import interview_pb2_grpc


INPUT_SIZE = 224
CAMERA_INDEX = 0


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    tensor = rgb.astype(np.float32) / 255.0
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
    except grpc.RpcError as exc:
        print("[ERROR] gRPC server is not running.")
        print(exc.code(), exc.details())
        return

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("[ERROR] Webcam open failed.")
        return

    print("[INFO] Local camera test started.")
    print("[INFO] Press q to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Failed to read webcam frame.")
            break

        tensor = preprocess_frame(frame)

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

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
