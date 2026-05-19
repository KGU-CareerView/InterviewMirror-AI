import time
from concurrent import futures
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


BASE_DIR = Path(__file__).resolve().parent.parent


def get_model_path() -> str:
    candidates = [
        Path(config.MODEL_PATH),
        BASE_DIR / "models" / "interview_model_v3.onnx",
        BASE_DIR / "models" / "emotion_model.onnx",
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        f"ONNX model not found. Checked: {[str(path) for path in candidates]}"
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

    def HealthCheck(self, request, context):
        question_client_status = "enabled" if self.question_client is not None else "disabled"

        return interview_pb2.HealthResponse(
            status="ok",
            model_path=MODEL_PATH,
            question_client_status=question_client_status,
        )

    def AnalyzeFrame(self, request, context):
        return self._analyze(request)

    def AnalyzeFrameStream(self, request_iterator, context):
        for request in request_iterator:
            yield self._analyze(request)

    def GenerateInitialQuestions(self, request, context):
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
                feedback="얼굴이 감지되지 않았습니다. 카메라 정면을 바라봐 주세요.",
                timestamp=request.timestamp,
                face_detected=False,
                bbox=request.bbox,
            )

        if not request.features:
            return self._error_response(
                request,
                feedback="전달받은 feature 데이터가 비어 있습니다.",
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
                "feature 개수와 tensor_shape가 맞지 않습니다. "
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
            bbox=request.bbox,
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
