import os
import time
import tempfile
from concurrent import futures
from pathlib import Path

import grpc
import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv

# gRPC 생성 파일들 (파일명이 interview_pb2.py 인지 확인 필수)
import interview_pb2
import interview_pb2_grpc
from labels import CLASS_NAMES

# 자막 관련 모듈
from subtitles.gemini_client import GeminiSubtitleClient
from subtitles.subtitle_formatter import subtitle_result_to_srt

# 1. 환경 변수 로드
load_dotenv()

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

# --- 서비스 구현 클래스 ---
class InterviewAIService(interview_pb2_grpc.InterviewAIServiceServicer):
    def __init__(self):
        print(f"[INFO] Loading ONNX model: {MODEL_PATH}")
        # CPU 환경에서 안정적인 실행을 위해 providers 지정
        self.session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
        # Gemini 자막 생성 클라이언트 로드
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
        temp_audio_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
                temp_audio.write(request.audio)
                temp_audio_path = Path(temp_audio.name)

            # GeminiSubtitleClient 내부에서 model="gemini-2.0-flash"를 사용하는지 꼭 확인하세요!
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
            print(f"[ERROR] Subtitle failed: {exc}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Subtitle generation failed: {exc}")
            return interview_pb2.SubtitleResponse(
                session_id=request.session_id,
                user_id=request.user_id,
            )
        finally:
            if temp_audio_path and temp_audio_path.exists():
                temp_audio_path.unlink()

    def _get_audio_suffix(self, audio_mime_type: str) -> str:
        mapping = {
            "audio/mpeg": ".mp3",
            "audio/wav": ".wav",
            "audio/mp4": ".m4a",
            "audio/webm": ".webm"
        }
        return mapping.get(audio_mime_type, ".wav")

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

        try:
            input_tensor = self._features_to_tensor(request.features, request.tensor_shape)
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
        except Exception as e:
            return self._error_response(request, feedback=f"추론 오류: {str(e)}")

    def _features_to_tensor(self, features, tensor_shape):
        shape = list(tensor_shape) if tensor_shape else DEFAULT_TENSOR_SHAPE
        expected_size = int(np.prod(shape))
        if len(features) != expected_size:
            raise ValueError(f"Feature size mismatch: expected {expected_size}, got {len(features)}")
        return np.array(features, dtype=np.float32).reshape(shape)

    def _softmax(self, logits):
        exps = np.exp(logits - np.max(logits))
        return exps / np.sum(exps)

    def _make_feedback(self, label: str, confidence: float) -> str:
        feedbacks = {
            "stable_confident": "표정이 안정적이고 자신감 있어 보입니다. 좋습니다!",
            "nervous_anxious": "조금 긴장하신 것 같네요. 천천히 호흡하며 말씀해 보세요.",
            "neutral": "차분한 상태입니다. 답변에 약간의 생기를 더해보세요."
        }
        return feedbacks.get(label, "분석 결과를 확인 중입니다.")

    def _error_response(self, request, feedback: str):
        return interview_pb2.AnalysisResponse(
            session_id=request.session_id,
            user_id=request.user_id,
            label="error",
            feedback=feedback,
            timestamp=request.timestamp,
            face_detected=False,
        )

# --- 서버 실행 함수 ---
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    interview_pb2_grpc.add_InterviewAIServiceServicer_to_server(InterviewAIService(), server)
    
    server.add_insecure_port("[::]:50051")
    server.start()
    print("[gRPC Server] Interview AI server started on port 50051")
    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    serve()
    