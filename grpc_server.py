import os
import time
from concurrent import futures

import cv2
import grpc
import numpy as np
import onnxruntime as ort

import interview_pb2
import interview_pb2_grpc
from face_cropper import FaceCropper
from labels import CLASS_NAMES

try:
    import config
except ImportError:
    config = None


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(filename: str) -> str:
    if os.path.isabs(filename):
        return filename
    return os.path.join(BASE_DIR, filename)


def get_model_path() -> str:
    # 기존 config.py에는 emotion_model.onnx로 되어 있지만,
    # 현재 업로드된 실제 모델 파일명은 interview_model_v3.onnx이므로 fallback 처리한다.
    candidates = []

    if config is not None and hasattr(config, "MODEL_PATH"):
        candidates.append(resolve_path(config.MODEL_PATH))

    candidates.append(resolve_path("interview_model_v3.onnx"))
    candidates.append(resolve_path("emotion_model.onnx"))

    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(f"ONNX model not found. Checked: {candidates}")


MODEL_PATH = get_model_path()
FACE_TASK_MODEL = resolve_path(
    getattr(config, "MEDIAPIPE_FACE_TASK_MODEL", "face_detector.task")
)
INPUT_SIZE = getattr(config, "INPUT_SIZE", 224)
MEAN = np.array(getattr(config, "MEAN", [0.485, 0.456, 0.406]), dtype=np.float32)
STD = np.array(getattr(config, "STD", [0.229, 0.224, 0.225]), dtype=np.float32)
MIN_DETECTION_CONFIDENCE = getattr(config, "MIN_DETECTION_CONFIDENCE", 0.6)
BBOX_MARGIN = getattr(config, "BBOX_MARGIN", 0.25)


class InterviewAIService(interview_pb2_grpc.InterviewAIServiceServicer):
    def __init__(self):
        print(f"[INFO] Loading ONNX model: {MODEL_PATH}")
        self.session = ort.InferenceSession(MODEL_PATH)
        self.input_name = self.session.get_inputs()[0].name

        print(f"[INFO] Loading MediaPipe face detector: {FACE_TASK_MODEL}")
        self.face_cropper = FaceCropper(
            model_path=FACE_TASK_MODEL,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            bbox_margin=BBOX_MARGIN,
        )

    def HealthCheck(self, request, context):
        return interview_pb2.HealthResponse(
            status="ok",
            model_path=MODEL_PATH,
        )

    def AnalyzeFrame(self, request, context):
        return self._analyze(request)

    def AnalyzeFrameStream(self, request_iterator, context):
        for request in request_iterator:
            yield self._analyze(request)

    def _analyze(self, request):
        frame = self._decode_image(request.image)

        if frame is None:
            return self._error_response(
                request,
                feedback="이미지 bytes를 OpenCV frame으로 변환하지 못했습니다.",
            )

        face_crop, bbox = self.face_cropper.get_largest_face(frame)

        if face_crop is None or bbox is None:
            return interview_pb2.AnalysisResponse(
                session_id=request.session_id,
                user_id=request.user_id,
                label="no_face",
                confidence=0.0,
                feedback="얼굴이 감지되지 않았습니다. 카메라 정면을 바라봐 주세요.",
                timestamp=request.timestamp,
                face_detected=False,
            )

        input_tensor = self._preprocess(face_crop)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        logits = outputs[0]

        probs = self._softmax(logits[0])
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)

        x1, y1, x2, y2 = bbox

        return interview_pb2.AnalysisResponse(
            session_id=request.session_id,
            user_id=request.user_id,
            label=label,
            confidence=confidence,
            feedback=self._make_feedback(label, confidence),
            timestamp=request.timestamp,
            face_detected=True,
            bbox=interview_pb2.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        )

    def _decode_image(self, image_bytes: bytes):
        if not image_bytes:
            return None

        np_arr = np.frombuffer(image_bytes, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    def _preprocess(self, face_bgr):
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (INPUT_SIZE, INPUT_SIZE))

        face = face_rgb.astype(np.float32) / 255.0
        face = (face - MEAN) / STD
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)

        return face.astype(np.float32)

    def _softmax(self, logits):
        logits = logits.astype(np.float32)
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        return exp / np.sum(exp)

    def _make_feedback(self, label: str, confidence: float) -> str:
        if label == "stable_confident":
            return "표정이 안정적이고 자신감 있어 보입니다. 현재 분위기를 유지해도 좋습니다."
        if label == "nervous_anxious":
            return "긴장감이 다소 보입니다. 시선을 정면에 두고 말의 속도를 조금 낮춰보세요."
        if label == "neutral":
            return "표정이 중립적입니다. 답변 중간에 자연스러운 미소를 조금 더하면 좋습니다."
        return "분석 결과를 확인했습니다."

    def _error_response(self, request, feedback: str):
        return interview_pb2.AnalysisResponse(
            session_id=request.session_id,
            user_id=request.user_id,
            label="error",
            confidence=0.0,
            feedback=feedback,
            timestamp=request.timestamp,
            face_detected=False,
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    interview_pb2_grpc.add_InterviewAIServiceServicer_to_server(
        InterviewAIService(),
        server,
    )

    server.add_insecure_port("[::]:50051")
    server.start()
    print("[gRPC Server] Interview AI gRPC server started on port 50051")

    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        print("[gRPC Server] stopping...")
        server.stop(0)


if __name__ == "__main__":
    serve()
