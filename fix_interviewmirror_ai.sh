#!/usr/bin/env bash
set -euo pipefail

mkdir -p questions
mkdir -p backup_before_fix

for f in grpc_server.py interview.proto grpc_client_test.py question_client_test.py questions/gemini_question_client.py questions/schemas.py; do
  if [ -f "$f" ]; then
    cp "$f" "backup_before_fix/$f.$(date +%Y%m%d_%H%M%S).bak" 2>/dev/null || true
  fi
done

cat > interview.proto <<'PYEOF'
syntax = "proto3";

package interview;

service InterviewAIService {
  rpc AnalyzeFrame (FeatureRequest) returns (AnalysisResponse);
  rpc AnalyzeFrameStream (stream FeatureRequest) returns (stream AnalysisResponse);
  rpc HealthCheck (HealthRequest) returns (HealthResponse);
  rpc GenerateInitialQuestions (InitialQuestionGenerateRequest) returns (InitialQuestionGenerateResponse);
  rpc GenerateFollowUpQuestion (FollowUpQuestionGenerateRequest) returns (FollowUpQuestionGenerateResponse);
}

message FeatureRequest {
  string session_id = 1;
  string user_id = 2;
  repeated int32 tensor_shape = 3;
  repeated float features = 4;
  int64 timestamp = 5;
  bool face_detected = 6;
  BoundingBox bbox = 7;
}

message BoundingBox {
  int32 x1 = 1;
  int32 y1 = 2;
  int32 x2 = 3;
  int32 y2 = 4;
}

message AnalysisResponse {
  string session_id = 1;
  string user_id = 2;
  string label = 3;
  float confidence = 4;
  string feedback = 5;
  int64 timestamp = 6;
  bool face_detected = 7;
  BoundingBox bbox = 8;
}

message HealthRequest {}

message HealthResponse {
  string status = 1;
  string model_path = 2;
  string question_client_status = 3;
}

message InitialQuestionGenerateRequest {
  string session_id = 1;
  string user_id = 2;
  string category = 3;
  string interview_type = 4;
  string difficulty = 5;
  int32 question_count = 6;
  int32 time_per_question = 7;
  string resume_text = 8;
  string language = 9;
}

message InitialQuestionGenerateResponse {
  string session_id = 1;
  string user_id = 2;
  repeated QuestionItem questions = 3;
}

message FollowUpQuestionGenerateRequest {
  string session_id = 1;
  string user_id = 2;
  string previous_question = 3;
  string answer = 4;
  string category = 5;
  string interview_type = 6;
  string difficulty = 7;
  string resume_text = 8;
  repeated InterviewTurn history = 9;
  string language = 10;
}

message FollowUpQuestionGenerateResponse {
  string session_id = 1;
  string user_id = 2;
  QuestionItem question = 3;
}

message QuestionItem {
  int32 index = 1;
  string question = 2;
  string tooltip = 3;
  string category = 4;
  string intent = 5;
  repeated string answer_keywords = 6;
}

message InterviewTurn {
  int32 index = 1;
  string question = 2;
  string answer = 3;
}
PYEOF

cat > questions/schemas.py <<'PYEOF'
from pydantic import BaseModel, Field, field_validator, model_validator


class QuestionItemModel(BaseModel):
    index: int = Field(description="Question index starting from 1.")
    question: str = Field(description="Generated interview question.")
    tooltip: str = Field(description="Short helpful tooltip for answering the question.")
    category: str = Field(description="Question category.")
    intent: str = Field(description="Why this question is being asked.")
    answer_keywords: list[str] = Field(description="Useful keywords or points the interviewee should include.")

    @field_validator("question", "tooltip", "category", "intent")
    @classmethod
    def validate_not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Text field must not be empty.")
        return value

    @field_validator("answer_keywords")
    @classmethod
    def validate_keywords(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item and item.strip()]
        if not cleaned:
            raise ValueError("answer_keywords must not be empty.")
        return cleaned

    @model_validator(mode="after")
    def validate_index(self) -> "QuestionItemModel":
        if self.index < 1:
            raise ValueError("index must start from 1.")
        return self


class QuestionGenerateResult(BaseModel):
    questions: list[QuestionItemModel] = Field(description="Generated interview questions with tooltips.")

    @field_validator("questions")
    @classmethod
    def validate_questions(cls, value: list[QuestionItemModel]) -> list[QuestionItemModel]:
        if not value:
            raise ValueError("questions must not be empty.")
        return value


class FollowUpQuestionGenerateResult(BaseModel):
    question: QuestionItemModel = Field(description="Generated follow-up interview question.")
PYEOF

cat > questions/gemini_question_client.py <<'PYEOF'
import os
from typing import Any

from dotenv import load_dotenv
from google import genai
from pydantic import ValidationError

from questions.schemas import (
    FollowUpQuestionGenerateResult,
    QuestionGenerateResult,
)

load_dotenv()


class GeminiQuestionClient:
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")

        self.client = genai.Client(api_key=self.api_key)

    def generate_initial_questions(
        self,
        category: str,
        interview_type: str,
        difficulty: str,
        question_count: int,
        time_per_question: int,
        resume_text: str,
        language: str,
    ) -> QuestionGenerateResult:
        question_count = self._normalize_question_count(question_count)
        prompt = self._build_initial_prompt(
            category=category,
            interview_type=interview_type,
            difficulty=difficulty,
            question_count=question_count,
            time_per_question=time_per_question,
            resume_text=resume_text,
            language=language,
        )
        return self._generate_json(prompt, QuestionGenerateResult)

    def generate_follow_up_question(
        self,
        previous_question: str,
        answer: str,
        category: str,
        interview_type: str,
        difficulty: str,
        resume_text: str,
        history: list[dict[str, Any]],
        language: str,
    ) -> FollowUpQuestionGenerateResult:
        prompt = self._build_follow_up_prompt(
            previous_question=previous_question,
            answer=answer,
            category=category,
            interview_type=interview_type,
            difficulty=difficulty,
            resume_text=resume_text,
            history=history,
            language=language,
        )
        return self._generate_json(prompt, FollowUpQuestionGenerateResult)

    def _generate_json(self, prompt: str, schema_model):
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": schema_model,
            },
        )

        parsed = getattr(response, "parsed", None)
        if parsed is not None:
            return parsed

        try:
            return schema_model.model_validate_json(response.text)
        except ValidationError as exc:
            raise ValueError(f"Invalid Gemini response: {exc}\nraw={response.text}") from exc

    def _normalize_question_count(self, question_count: int) -> int:
        if question_count <= 0:
            return 3
        if question_count > 10:
            return 10
        return question_count

    def _build_initial_prompt(
        self,
        category: str,
        interview_type: str,
        difficulty: str,
        question_count: int,
        time_per_question: int,
        resume_text: str,
        language: str,
    ) -> str:
        category = category.strip() or "종합"
        interview_type = interview_type.strip() or "기술 면접"
        difficulty = difficulty.strip() or "normal"
        resume_text = resume_text.strip() or "제공된 이력서/자기소개서 내용 없음"
        language = language.strip() or "ko-KR"
        time_per_question = time_per_question or 60

        return f"""
너는 AI 모의면접 서비스의 면접 질문 생성기다.

면접 조건:
- 카테고리: {category}
- 면접 유형: {interview_type}
- 난이도: {difficulty}
- 질문 수: {question_count}
- 질문당 답변 시간: {time_per_question}초
- 응답 언어: {language}

이력서/자기소개서/포트폴리오 내용:
{resume_text}

생성 규칙:
1. 실제 면접에서 물어볼 법한 자연스러운 질문을 만든다.
2. resume_text가 있으면 개인화 질문을 우선 생성한다.
3. 질문은 서로 의미가 겹치지 않게 만든다.
4. tooltip은 사용자가 답변 방향을 잡을 수 있게 짧게 작성한다.
5. category는 인성, 직무, 기술, 프로젝트, 경험, 꼬리질문, 종합 중 하나에 가깝게 작성한다.
6. intent는 이 질문으로 무엇을 확인하려는지 설명한다.
7. answer_keywords는 3~6개 작성한다.
8. 반드시 JSON schema에 맞게 반환한다.
""".strip()

    def _build_follow_up_prompt(
        self,
        previous_question: str,
        answer: str,
        category: str,
        interview_type: str,
        difficulty: str,
        resume_text: str,
        history: list[dict[str, Any]],
        language: str,
    ) -> str:
        return f"""
너는 AI 모의면접 서비스의 꼬리질문 생성기다.

면접 조건:
- 카테고리: {category or '종합'}
- 면접 유형: {interview_type or '기술 면접'}
- 난이도: {difficulty or 'normal'}
- 응답 언어: {language or 'ko-KR'}

이력서/자기소개서/포트폴리오 내용:
{resume_text or '제공된 내용 없음'}

이전 면접 이력:
{history}

직전 질문:
{previous_question}

지원자 답변:
{answer}

생성 규칙:
1. 지원자 답변에서 더 검증할 만한 부분을 하나 골라 꼬리질문 1개를 만든다.
2. 너무 공격적이지 않게, 실제 면접관처럼 자연스럽게 질문한다.
3. tooltip은 답변 방향을 짧게 알려준다.
4. category는 가능하면 꼬리질문으로 작성한다.
5. answer_keywords는 3~6개 작성한다.
6. 반드시 JSON schema에 맞게 반환한다.
""".strip()
PYEOF

cat > grpc_server.py <<'PYEOF'
import os
import time
from concurrent import futures

import grpc
import numpy as np
import onnxruntime as ort

import interview_pb2
import interview_pb2_grpc
from labels import CLASS_NAMES

try:
    import config
except ImportError:
    config = None

try:
    from questions.gemini_question_client import GeminiQuestionClient
except Exception as exc:
    GeminiQuestionClient = None
    QUESTION_CLIENT_IMPORT_ERROR = exc
else:
    QUESTION_CLIENT_IMPORT_ERROR = None

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
DEFAULT_INPUT_SIZE = getattr(config, "INPUT_SIZE", 224) if config is not None else 224
DEFAULT_TENSOR_SHAPE = [1, 3, DEFAULT_INPUT_SIZE, DEFAULT_INPUT_SIZE]


class InterviewAIService(interview_pb2_grpc.InterviewAIServiceServicer):
    def __init__(self):
        print(f"[INFO] Loading ONNX model: {MODEL_PATH}")
        self.session = ort.InferenceSession(MODEL_PATH)
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
                questions=[self._to_proto_question_item(item) for item in result.questions],
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
                {"index": turn.index, "question": turn.question, "answer": turn.answer}
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
            answer_keywords=item.answer_keywords,
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
            return self._error_response(request, feedback="전달받은 feature 데이터가 비어 있습니다.")

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
    interview_pb2_grpc.add_InterviewAIServiceServicer_to_server(InterviewAIService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()

    print("[gRPC Server] Interview AI gRPC server started on port 50051")
    print("[gRPC Server] Server receives features only. No image bytes / MediaPipe / OpenCV processing here.")
    print("[gRPC Server] GenerateInitialQuestions RPC is available.")
    print("[gRPC Server] GenerateFollowUpQuestion RPC is available.")

    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        print("[gRPC Server] stopping...")
        server.stop(0)


if __name__ == "__main__":
    serve()
PYEOF

cat > grpc_client_test.py <<'PYEOF'
import time

import grpc
import numpy as np

import interview_pb2
import interview_pb2_grpc


def make_dummy_request():
    shape = [1, 3, 224, 224]
    features = np.zeros(shape, dtype=np.float32).reshape(-1).tolist()
    return interview_pb2.FeatureRequest(
        session_id="test-session-001",
        user_id="test-user-001",
        tensor_shape=shape,
        features=features,
        timestamp=int(time.time() * 1000),
        face_detected=True,
        bbox=interview_pb2.BoundingBox(x1=0, y1=0, x2=224, y2=224),
    )


def main():
    channel = grpc.insecure_channel("127.0.0.1:50051")
    stub = interview_pb2_grpc.InterviewAIServiceStub(channel)

    health = stub.HealthCheck(interview_pb2.HealthRequest())
    print("[HEALTH]", health.status)
    print("[MODEL]", health.model_path)
    print("[QUESTION CLIENT]", health.question_client_status)

    response = stub.AnalyzeFrame(make_dummy_request())
    print("[ANALYSIS]")
    print("label=", response.label)
    print("confidence=", response.confidence)
    print("feedback=", response.feedback)


if __name__ == "__main__":
    main()
PYEOF

cat > question_client_test.py <<'PYEOF'
import grpc

import interview_pb2
import interview_pb2_grpc


def test_initial_questions(stub):
    response = stub.GenerateInitialQuestions(
        interview_pb2.InitialQuestionGenerateRequest(
            session_id="test-session-001",
            user_id="test-user-001",
            category="백엔드",
            interview_type="기술 면접",
            difficulty="normal",
            question_count=3,
            time_per_question=60,
            resume_text="""
Spring Boot와 MySQL을 활용해 게시판 프로젝트를 구현했습니다.
JWT 기반 로그인, 게시글 CRUD, 댓글 기능, 파일 업로드 기능을 맡았습니다.
gRPC 기반 AI 서버 연동 경험이 있습니다.
""",
            language="ko-KR",
        )
    )

    print("\n========== INITIAL QUESTIONS ==========")
    for item in response.questions:
        print("=" * 60)
        print(f"[{item.index}] {item.question}")
        print(f"카테고리: {item.category}")
        print(f"의도: {item.intent}")
        print(f"툴팁: {item.tooltip}")
        print(f"키워드: {list(item.answer_keywords)}")


def test_follow_up_question(stub):
    response = stub.GenerateFollowUpQuestion(
        interview_pb2.FollowUpQuestionGenerateRequest(
            session_id="test-session-001",
            user_id="test-user-001",
            previous_question="gRPC 기반 AI 서버 연동 경험에 대해 설명해 주세요.",
            answer="""
REST API보다 실시간 스트리밍에 유리하다고 판단해서 gRPC를 사용했습니다.
프론트에서 추출한 특징점 데이터를 서버로 보내고, 서버는 ONNX 모델로 추론한 뒤 결과를 반환하는 구조였습니다.
""",
            category="백엔드",
            interview_type="기술 면접",
            difficulty="normal",
            resume_text="Spring Boot와 MySQL을 활용해 게시판 프로젝트를 구현했습니다.",
            history=[
                interview_pb2.InterviewTurn(
                    index=1,
                    question="본인이 맡았던 백엔드 프로젝트 경험을 설명해 주세요.",
                    answer="게시판 프로젝트에서 로그인, CRUD, 파일 업로드 기능을 구현했습니다.",
                )
            ],
            language="ko-KR",
        )
    )

    item = response.question
    print("\n========== FOLLOW-UP QUESTION ==========")
    print("=" * 60)
    print(f"[{item.index}] {item.question}")
    print(f"카테고리: {item.category}")
    print(f"의도: {item.intent}")
    print(f"툴팁: {item.tooltip}")
    print(f"키워드: {list(item.answer_keywords)}")


def main():
    channel = grpc.insecure_channel("127.0.0.1:50051")
    stub = interview_pb2_grpc.InterviewAIServiceStub(channel)

    health = stub.HealthCheck(interview_pb2.HealthRequest())
    print("[HEALTH]", health.status)
    print("[MODEL]", health.model_path)
    print("[QUESTION CLIENT]", health.question_client_status)

    test_initial_questions(stub)
    test_follow_up_question(stub)


if __name__ == "__main__":
    main()
PYEOF

python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. interview.proto

python -m py_compile grpc_server.py grpc_client_test.py question_client_test.py questions/gemini_question_client.py questions/schemas.py

echo "✅ Fix applied. Next: python grpc_server.py"
