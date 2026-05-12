import os
import time
from concurrent import futures

import grpc
import numpy as np
import onnxruntime as ort

import interview_pb2
import interview_pb2_grpc
from labels import CLASS_NAMES

import tempfile
from pathlib import Path

from subtitles.gemini_client import GeminiSubtitleClient
from subtitles.subtitle_formatter import subtitle_result_to_srt

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
DEFAULT_INPUT_SIZE = getattr(config, "INPUT_SIZE", 224)
DEFAULT_TENSOR_SHAPE = [1, 3, DEFAULT_INPUT_SIZE, DEFAULT_INPUT_SIZE]


class InterviewAIService(interview_pb2_grpc.InterviewAIServiceServicer):
    def __init__(self):
        print(f"[INFO] Loading ONNX model: {MODEL_PATH}")
        self.session = ort.InferenceSession(MODEL_PATH)
        self.input_name = self.session.get_inputs()[0].name

        # Gemini 자막 생성 클라이언트
        self.subtitle_client = GeminiSubtitleClient()

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

    def GenerateSubtitles(self, request, context):
        suffix = self._get_audio_suffix(request.audio_mime_type)

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
                temp_audio.write(request.audio)
                temp_audio_path = Path(temp_audio.name)

            result = self.subtitle_client.generate_subtitles(
                audio_path=temp_audio_path,
                language_hint=request.language_hint or "ko-KR",
            )

            srt = subtitle_result_to_srt(result)

            return interview_pb2.SubtitleResponse(
                session_id=request.session_id,
                user_id=request.user_id,
                language=result.language,
                summary=result.summary,
                segments=[
                    interview_pb2.SubtitleSegmentMessage(
                        index=segment.index,
                        start_ms=segment.start_ms,
                        end_ms=segment.end_ms,
                        text=segment.text,
                    )
                    for segment in result.segments
                ],
                srt=srt,
            )

        except Exception as exc:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Subtitle generation failed: {exc}")

            return interview_pb2.SubtitleResponse(
                session_id=request.session_id,
                user_id=request.user_id,
            )

        finally:
            if "temp_audio_path" in locals() and temp_audio_path.exists():
                temp_audio_path.unlink()

    def _get_audio_suffix(self, audio_mime_type: str) -> str:
        if audio_mime_type == "audio/mpeg":
            return ".mp3"
        if audio_mime_type == "audio/wav":
            return ".wav"
        if audio_mime_type == "audio/mp4":
            return ".m4a"
        if audio_mime_type == "audio/webm":
            return ".webm"
        return ".wav"

    def _analyze(self, request):
        if not request.face_detected:
            return interview_pb2.AnalysisResponse(
                session_id=request.session_id,
                user_id=request.user_id,
                label="no_face",
                confidence=0.0,
                feedback="얼굴이 감지되지 않았습니다. 카메라 정면을 바라봐 주세요.",
                timestamp=request.timestamp,
                face_detected=False,
            )

        if not request.features:
            return self._error_response(
                request,
                feedback="전달받은 feature 데이터가 비어 있습니다.",
            )

        try:
            input_tensor = self._features_to_tensor(request.features, request.tensor_shape)
        except ValueError as exc:
            return self._error_response(request, feedback=str(exc))

        outputs = self.session.run(None, {self.input_name: input_tensor})
        logits = outputs[0]

        probs = self._softmax(logits[0])
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)

        return interview_pb2.AnalysisResponse(
            session_id=request.session_id,
            user_id=request.user_id,
            label=label,
            confidence=confidence,
            feedback=self._make_feedback(label, confidence),
            timestamp=request.timestamp,
            face_detected=True,
            bbox=request.bbox,
        )

    def _features_to_tensor(self, features, tensor_shape):
        shape = list(tensor_shape) if tensor_shape else DEFAULT_TENSOR_SHAPE
        expected_size = int(np.prod(shape))

        if len(features) != expected_size:
            raise ValueError(
                f"feature 개수와 tensor_shape가 맞지 않습니다. "
                f"features={len(features)}, shape={shape}, expected={expected_size}"
            )

        return np.array(features, dtype=np.float32).reshape(shape)

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
    print("[gRPC Server] Server receives features only. No image bytes / MediaPipe / OpenCV processing here.")

    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        print("[gRPC Server] stopping...")
        server.stop(0)


if __name__ == "__main__":
    serve()
