# InterviewMirror AI gRPC server
import os
import tempfile
import time
from concurrent import futures
from datetime import datetime
from pathlib import Path

import grpc
import numpy as np
import onnxruntime as ort

from app import config
from app.labels import CLASS_NAMES
from generated import interview_pb2
from generated import interview_pb2_grpc

try:
    from app.questions.gemini_question_client import GeminiQuestionClient
except Exception as exc:
    GeminiQuestionClient = None
    QUESTION_CLIENT_IMPORT_ERROR = exc
else:
    QUESTION_CLIENT_IMPORT_ERROR = None

try:
    from app.subtitles.gemini_client import GeminiSubtitleClient
    from app.subtitles.subtitle_formatter import subtitle_result_to_srt
except Exception as exc:
    GeminiSubtitleClient = None
    subtitle_result_to_srt = None
    SUBTITLE_CLIENT_IMPORT_ERROR = exc
else:
    SUBTITLE_CLIENT_IMPORT_ERROR = None

try:
    from app.voice.voice_tone_analyzer import VoiceToneAnalyzer
except Exception as exc:
    VoiceToneAnalyzer = None
    VOICE_TONE_ANALYZER_IMPORT_ERROR = exc
else:
    VOICE_TONE_ANALYZER_IMPORT_ERROR = None


BASE_DIR = Path(__file__).resolve().parent.parent


def get_model_path() -> str:
    candidates = []

    for path in [
        Path(config.MODEL_PATH),
        BASE_DIR / "models" / "interview_model_v3.onnx",
        BASE_DIR / "models" / "emotion_model.onnx",
    ]:
        if path not in candidates:
            candidates.append(path)

    for path in sorted((BASE_DIR / "models").glob("*.onnx")):
        if path not in candidates:
            candidates.append(path)

    for path in candidates:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        "ONNX model not found. "
        f"Checked: {[str(path) for path in candidates]}"
    )


MODEL_PATH = get_model_path()
DEFAULT_INPUT_SIZE = getattr(config, "INPUT_SIZE", 224)
DEFAULT_TENSOR_SHAPE = [1, 3, DEFAULT_INPUT_SIZE, DEFAULT_INPUT_SIZE]


class InterviewAIService(interview_pb2_grpc.InterviewAIServiceServicer):
    def __init__(self):
        print(f"[INFO] Loading ONNX model: {MODEL_PATH}")

        self.session = ort.InferenceSession(str(MODEL_PATH))
        self.input_name = self.session.get_inputs()[0].name

        self.question_client = self._init_question_client()
        self.subtitle_client = self._init_subtitle_client()
        self.voice_tone_analyzer = self._init_voice_tone_analyzer()

    def _init_question_client(self):
        if GeminiQuestionClient is None:
            print(f"[WARN] Question client import failed: {QUESTION_CLIENT_IMPORT_ERROR}")
            return None

        try:
            client = GeminiQuestionClient()
            print("[INFO] Gemini question client enabled")
            return client
        except Exception as exc:
            print(f"[WARN] Gemini question client disabled: {exc}")
            return None

    def _init_subtitle_client(self):
        if GeminiSubtitleClient is None:
            print(f"[WARN] Subtitle client import failed: {SUBTITLE_CLIENT_IMPORT_ERROR}")
            return None

        try:
            client = GeminiSubtitleClient()
            print("[INFO] Gemini subtitle client enabled")
            return client
        except Exception as exc:
            print(f"[WARN] Gemini subtitle client disabled: {exc}")
            return None

    def _init_voice_tone_analyzer(self):
        if VoiceToneAnalyzer is None:
            print(f"[WARN] Voice tone analyzer import failed: {VOICE_TONE_ANALYZER_IMPORT_ERROR}")
            return None

        try:
            analyzer = VoiceToneAnalyzer()
            print("[INFO] Voice tone analyzer enabled")
            return analyzer
        except Exception as exc:
            print(f"[WARN] Voice tone analyzer disabled: {exc}")
            return None

    def _log_request_header(self, rpc_name: str):
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print("\n" + "=" * 80)
        print(f"[REQUEST] {rpc_name} | {now}")
        print("=" * 80)

    def _log_feature_request(self, rpc_name: str, request):
        self._log_request_header(rpc_name)
        bbox = request.bbox

        print(f"session_id : {request.session_id}")
        print(f"user_id : {request.user_id}")
        print(f"timestamp : {request.timestamp}")
        print(f"face_detected : {request.face_detected}")
        print(f"bbox : x1={bbox.x1}, y1={bbox.y1}, x2={bbox.x2}, y2={bbox.y2}")
        print(f"tensor_shape : {list(request.tensor_shape)}")
        print(f"features_len : {len(request.features)}")

    def _log_audio_request(self, rpc_name: str, request):
        self._log_request_header(rpc_name)
        print(f"session_id : {request.session_id}")
        print(f"user_id : {request.user_id}")
        print(f"audio_mime_type : {request.audio_mime_type}")
        print(f"language_hint : {request.language_hint}")
        print(f"audio_size_bytes : {len(request.audio)}")

    def _get_audio_suffix(self, audio_mime_type: str) -> str:
        mime = (audio_mime_type or "").lower()

        if "wav" in mime:
            return ".wav"
        if "mpeg" in mime or "mp3" in mime:
            return ".mp3"
        if "webm" in mime:
            return ".webm"
        if "ogg" in mime:
            return ".ogg"
        if "m4a" in mime or "mp4" in mime:
            return ".m4a"

        return ".wav"

    def HealthCheck(self, request, context):
        return interview_pb2.HealthResponse(
            status="ok",
            model_path=MODEL_PATH,
            question_client_status="enabled" if self.question_client else "disabled",
            subtitle_client_status="enabled" if self.subtitle_client else "disabled",
            voice_tone_analyzer_status="enabled" if self.voice_tone_analyzer else "disabled",
            report_client_status="disabled",
        )

    def AnalyzeFrame(self, request, context):
        self._log_feature_request("AnalyzeFrame", request)
        return self._analyze(request)

    def AnalyzeFrameStream(self, request_iterator, context):
        for request in request_iterator:
            self._log_feature_request("AnalyzeFrameStream", request)
            yield self._analyze(request)

    def GenerateInitialQuestions(self, request, context):
        self._log_request_header("GenerateInitialQuestions")

        if self.question_client is None:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("Question generation is disabled. Set GEMINI_API_KEY in .env.")
            return interview_pb2.InitialQuestionGenerateResponse(
                session_id=request.session_id,
                user_id=request.user_id,
                questions=[],
            )

        try:
            result = self.question_client.generate_initial_questions(
                category=request.category,
                interview_type=request.interview_type,
                difficulty=request.difficulty,
                question_count=request.question_count,
                time_per_question=request.time_per_question,
                resume_text=request.resume_text,
                language=request.language,
            )

            return interview_pb2.InitialQuestionGenerateResponse(
                session_id=request.session_id,
                user_id=request.user_id,
                questions=[
                    self._to_proto_question_item(item)
                    for item in result.questions
                ],
            )

        except Exception as exc:
            print(f"[ERROR] Initial question generation failed: {exc}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Initial question generation failed: {exc}")

            return interview_pb2.InitialQuestionGenerateResponse(
                session_id=request.session_id,
                user_id=request.user_id,
                questions=[],
            )

    def GenerateFollowUpQuestion(self, request, context):
        self._log_request_header("GenerateFollowUpQuestion")

        if self.question_client is None:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("Question generation is disabled. Set GEMINI_API_KEY in .env.")
            return interview_pb2.FollowUpQuestionGenerateResponse(
                session_id=request.session_id,
                user_id=request.user_id,
            )

        try:
            history = [
                {
                    "index": turn.index,
                    "question": turn.question,
                    "answer": turn.answer,
                }
                for turn in request.history
            ]

            result = self.question_client.generate_follow_up_question(
                previous_question=request.previous_question,
                answer=request.answer,
                category=request.category,
                interview_type=request.interview_type,
                difficulty=request.difficulty,
                resume_text=request.resume_text,
                history=history,
                language=request.language,
            )

            return interview_pb2.FollowUpQuestionGenerateResponse(
                session_id=request.session_id,
                user_id=request.user_id,
                question=self._to_proto_question_item(result.question),
            )

        except Exception as exc:
            print(f"[ERROR] Follow-up question generation failed: {exc}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Follow-up question generation failed: {exc}")

            return interview_pb2.FollowUpQuestionGenerateResponse(
                session_id=request.session_id,
                user_id=request.user_id,
            )

    def GenerateSubtitles(self, request, context):
        self._log_audio_request("GenerateSubtitles", request)

        if self.subtitle_client is None:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("Subtitle generation is disabled. Set GEMINI_API_KEY in .env.")
            return interview_pb2.SubtitleResponse(
                session_id=request.session_id,
                user_id=request.user_id,
                language=request.language_hint or "ko-KR",
                summary="",
                segments=[],
                srt="",
            )

        if not request.audio:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Audio data is empty.")
            return interview_pb2.SubtitleResponse(
                session_id=request.session_id,
                user_id=request.user_id,
                language=request.language_hint or "ko-KR",
                summary="",
                segments=[],
                srt="",
            )

        suffix = self._get_audio_suffix(request.audio_mime_type)
        temp_audio_path = None

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
            print(f"[ERROR] Subtitle generation failed: {exc}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Subtitle generation failed: {exc}")

            return interview_pb2.SubtitleResponse(
                session_id=request.session_id,
                user_id=request.user_id,
                language=request.language_hint or "ko-KR",
                summary="",
                segments=[],
                srt="",
            )

        finally:
            if temp_audio_path is not None:
                try:
                    temp_audio_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def AnalyzeVoiceTone(self, request, context):
        self._log_audio_request("AnalyzeVoiceTone", request)

        if self.voice_tone_analyzer is None:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("Voice tone analyzer is disabled.")

            return interview_pb2.VoiceToneAnalysisResponse(
                session_id=request.session_id,
                user_id=request.user_id,
                feedback="Voice tone analyzer is disabled.",
            )

        if not request.audio:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("Audio data is empty.")

            return interview_pb2.VoiceToneAnalysisResponse(
                session_id=request.session_id,
                user_id=request.user_id,
                feedback="Audio data is empty.",
            )

        suffix = self._get_audio_suffix(request.audio_mime_type)
        temp_audio_path = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_audio:
                temp_audio.write(request.audio)
                temp_audio_path = Path(temp_audio.name)

            result = self.voice_tone_analyzer.analyze(temp_audio_path)

            return interview_pb2.VoiceToneAnalysisResponse(
                session_id=request.session_id,
                user_id=request.user_id,
                pitch_mean=result.pitch_mean,
                pitch_std=result.pitch_std,
                pitch_stability=result.pitch_stability,
                energy_mean=result.energy_mean,
                energy_std=result.energy_std,
                energy_stability=result.energy_stability,
                pause_ratio=result.pause_ratio,
                speech_duration_sec=result.speech_duration_sec,
                total_duration_sec=result.total_duration_sec,
                overall_stability_score=result.overall_stability_score,
                feedback=result.feedback,
            )

        except Exception as exc:
            print(f"[ERROR] Voice tone analysis failed: {exc}")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Voice tone analysis failed: {exc}")

            return interview_pb2.VoiceToneAnalysisResponse(
                session_id=request.session_id,
                user_id=request.user_id,
                feedback="Voice tone analysis failed.",
            )

        finally:
            if temp_audio_path is not None:
                try:
                    temp_audio_path.unlink(missing_ok=True)
                except Exception:
                    pass

    def GenerateFinalReport(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("GenerateFinalReport is temporarily disabled in stable recovery build.")
        return interview_pb2.FinalReportResponse(
            session_id=request.session_id,
            user_id=request.user_id,
            overall_summary="Final report generation is temporarily disabled.",
            overall_score=0.0,
            content_score=0.0,
            voice_score=0.0,
            expression_score=0.0,
            strengths=[],
            weaknesses=[],
            time_based_insights=[],
            final_advice="Restore stable server first, then enable final report generation.",
        )

    def _to_proto_question_item(self, item):
        return interview_pb2.QuestionItem(
            index=item.index,
            question=item.question,
            tooltip=item.tooltip,
            category=item.category,
            intent=item.intent,
            answer_keywords=list(item.answer_keywords),
        )

    def _analyze(self, request):
        if not request.face_detected:
            return interview_pb2.AnalysisResponse(
                session_id=request.session_id,
                user_id=request.user_id,
                label="no_face",
                confidence=0.0,
                feedback="No face detected. Please look at the camera.",
                timestamp=request.timestamp,
                face_detected=False,
                bbox=request.bbox,
            )

        if not request.features:
            return self._error_response(
                request,
                feedback="Feature data is empty.",
            )

        try:
            input_tensor = self._features_to_tensor(
                request.features,
                request.tensor_shape,
            )
        except ValueError as exc:
            return self._error_response(request, feedback=str(exc))

        outputs = self.session.run(None, {self.input_name: input_tensor})
        logits = outputs[0]
        probs = self._softmax(logits[0])

        nervous_prob = float(probs[0])
        neutral_prob = float(probs[1])
        confident_prob = float(probs[2])

        expression_score = (
            nervous_prob * 35.0
            + neutral_prob * 72.0
            + confident_prob * 95.0
        )

        print(f"probs:{probs}", flush=True)
        print(f"expression_score:{expression_score:.2f}", flush=True)

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        label = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)

        print("[MODEL DEBUG] logits:", logits[0].tolist())
        print("[MODEL DEBUG] probs:", probs.tolist())
        print("[MODEL DEBUG] pred_idx:", pred_idx)
        print("[MODEL DEBUG] label:", label)

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
                "Feature count does not match tensor shape. "
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
            return "Your facial expression looks stable and confident."

        if label == "nervous_anxious":
            return "Some nervousness is detected. Try to slow down and look forward."

        if label == "neutral":
            return "Your expression is neutral. Try adding a natural smile while answering."

        return "Analysis completed."

    def _error_response(self, request, feedback: str):
        return interview_pb2.AnalysisResponse(
            session_id=request.session_id,
            user_id=request.user_id,
            label="error",
            confidence=0.0,
            feedback=feedback,
            timestamp=request.timestamp,
            face_detected=False,
            bbox=request.bbox,
        )


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

    interview_pb2_grpc.add_InterviewAIServiceServicer_to_server(
        InterviewAIService(),
        server,
    )

    host = os.getenv("GRPC_HOST", "0.0.0.0")
    port = int(os.getenv("GRPC_PORT", "50051"))
    address = f"{host}:{port}"

    bound_port = server.add_insecure_port(address)
    if bound_port == 0:
        raise RuntimeError(f"Failed to bind gRPC server to {address}")

    server.start()

    print(f"[gRPC Server] Interview AI gRPC server started on {address}")

    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        print("[gRPC Server] stopping...")
        server.stop(0)


if __name__ == "__main__":
    serve()
