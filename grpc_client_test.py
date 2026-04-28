import time

import cv2
import grpc
import numpy as np

import interview_pb2
import interview_pb2_grpc
from face_cropper import FaceCropper

try:
    import config
except ImportError:
    config = None


INPUT_SIZE = getattr(config, "INPUT_SIZE", 224)
MEAN = np.array(getattr(config, "MEAN", [0.485, 0.456, 0.406]), dtype=np.float32)
STD = np.array(getattr(config, "STD", [0.229, 0.224, 0.225]), dtype=np.float32)
CAMERA_INDEX = getattr(config, "CAMERA_INDEX", 0)
MIN_DETECTION_CONFIDENCE = getattr(config, "MIN_DETECTION_CONFIDENCE", 0.6)
BBOX_MARGIN = getattr(config, "BBOX_MARGIN", 0.25)
FACE_TASK_MODEL = getattr(config, "MEDIAPIPE_FACE_TASK_MODEL", "face_detector.task")


def preprocess_face_to_feature(face_bgr):
    """
    client/front에서 수행되는 OpenCV 전처리.
    server는 이 결과만 받아서 ONNX 추론만 수행한다.
    """
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (INPUT_SIZE, INPUT_SIZE))

    face = face_rgb.astype(np.float32) / 255.0
    face = (face - MEAN) / STD
    face = np.transpose(face, (2, 0, 1))
    face = np.expand_dims(face, axis=0)

    return face.astype(np.float32)


def main():
    channel = grpc.insecure_channel("127.0.0.1:50051")
    stub = interview_pb2_grpc.InterviewAIServiceStub(channel)

    health = stub.HealthCheck(interview_pb2.HealthRequest())
    print("[HEALTH]", health.status, health.model_path)

    face_cropper = FaceCropper(
        model_path=FACE_TASK_MODEL,
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        bbox_margin=BBOX_MARGIN,
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Webcam open failed")
        face_cropper.close()
        return

    print("[INFO] Press q to quit")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = int(time.time() * 1000)
            face_crop, bbox = face_cropper.get_largest_face(frame)

            if face_crop is None or bbox is None:
                request = interview_pb2.FeatureRequest(
                    session_id="test-session-001",
                    user_id="test-user-001",
                    timestamp=timestamp,
                    face_detected=False,
                )
            else:
                input_tensor = preprocess_face_to_feature(face_crop)
                x1, y1, x2, y2 = bbox

                request = interview_pb2.FeatureRequest(
                    session_id="test-session-001",
                    user_id="test-user-001",
                    tensor_shape=list(input_tensor.shape),
                    features=input_tensor.flatten().tolist(),
                    timestamp=timestamp,
                    face_detected=True,
                    bbox=interview_pb2.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )

            response = stub.AnalyzeFrame(request)
            print(
                f"label={response.label}, "
                f"confidence={response.confidence:.4f}, "
                f"feedback={response.feedback}"
            )

            if response.face_detected:
                x1, y1, x2, y2 = (
                    response.bbox.x1,
                    response.bbox.y1,
                    response.bbox.x2,
                    response.bbox.y2,
                )
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

    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_cropper.close()


if __name__ == "__main__":
    main()
