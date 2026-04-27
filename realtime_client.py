import cv2
import numpy as np
import onnxruntime as ort

from mediapipe.python.solutions import face_detection


MODEL_PATH = "interview_model_v3.onnx"
LABELS = ["Stable", "Nervous", "Neutral"]


def load_model():
    print("[INFO] Loading model...")
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    return session, input_name


face_detector = face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)


def detect_face(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_detector.process(rgb)

    if not result.detections:
        return None

    det = result.detections[0]
    bbox = det.location_data.relative_bounding_box

    h, w, _ = frame.shape

    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    bw = int(bbox.width * w)
    bh = int(bbox.height * h)

    x = max(0, x)
    y = max(0, y)
    bw = min(bw, w - x)
    bh = min(bh, h - y)

    return x, y, bw, bh


def preprocess_frame(frame, face_bbox):
    x, y, w, h = face_bbox

    face = frame[y:y+h, x:x+w]

    if face.size == 0:
        return None

    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))

    face = face.astype(np.float32) / 255.0
    face = (face - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    face = np.transpose(face, (2, 0, 1))
    face = np.expand_dims(face, axis=0)

    return face.astype(np.float32)


def main():
    session, input_name = load_model()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Webcam open failed")
        return

    print("[INFO] Webcam started")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("[ERROR] Failed to read frame")
            break

        face_box = detect_face(frame)

        if face_box is not None:
            x, y, w, h = face_box

            input_data = preprocess_frame(frame, face_box)

            if input_data is not None:
                outputs = session.run(None, {input_name: input_data})
                logits = outputs[0]

                pred_idx = int(np.argmax(logits, axis=1)[0])
                label = LABELS[pred_idx]

                score = float(np.max(logits))

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{label} {score:.2f}",
                    (x, max(30, y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("Interview Emotion AI", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
