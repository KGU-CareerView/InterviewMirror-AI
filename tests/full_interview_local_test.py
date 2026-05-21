import argparse
import tempfile
import time
from pathlib import Path

import cv2
import grpc
import numpy as np
import sounddevice as sd
import soundfile as sf

from app import config
from generated import interview_pb2
from generated import interview_pb2_grpc


SERVER_ADDRESS = "127.0.0.1:50051"
SAMPLE_RATE = 16000

MEAN = np.array(getattr(config, "MEAN", [0.485, 0.456, 0.406]), dtype=np.float32)
STD = np.array(getattr(config, "STD", [0.229, 0.224, 0.225]), dtype=np.float32)
INPUT_SIZE = getattr(config, "INPUT_SIZE", 224)


def expression_score_from_response(response) -> float:
    if response.label == "stable_confident":
        base = 90.0
    elif response.label == "neutral":
        base = 70.0
    elif response.label == "nervous_anxious":
        base = 45.0
    elif response.label == "no_face":
        base = 20.0
    else:
        base = 50.0

    confidence_bonus = float(response.confidence) * 10.0

    return max(0.0, min(100.0, base + confidence_bonus))


def preprocess_frame(frame):
    resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)

    return arr.astype(np.float32)


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


def make_frame_request(frame, session_id, user_id):
    tensor = preprocess_frame(frame)
    face_detected, bbox = detect_face(frame)

    x1, y1, x2, y2 = bbox

    return interview_pb2.FeatureRequest(
        session_id=session_id,
        user_id=user_id,
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


def collect_expression_scores(stub, duration_sec, session_id, user_id):
    print(f"[INFO] Collecting facial expression scores for {duration_sec} seconds...")

    cap = cv2.VideoCapture(getattr(config, "CAMERA_INDEX", 0))

    if not cap.isOpened():
        print("[WARN] Cannot open camera. Expression score will be skipped.")
        return [], 0.0, "no_camera"

    scores = []
    labels = []

    start = time.time()
    last_sent = 0.0

    while time.time() - start < duration_sec:
        ret, frame = cap.read()

        if not ret:
            continue

        cv2.imshow("InterviewMirror Full Interview Test", frame)

        if time.time() - last_sent >= 0.7:
            request = make_frame_request(frame, session_id, user_id)

            try:
                response = stub.AnalyzeFrame(request, timeout=5)
                score = expression_score_from_response(response)

                scores.append(
                    {
                        "timestamp": int(time.time() * 1000),
                        "score": score,
                        "label": response.label,
                        "feedback": response.feedback,
                    }
                )
                labels.append(response.label)

                print(
                    "[EXPRESSION]",
                    "label=", response.label,
                    "confidence=", round(response.confidence, 3),
                    "score=", round(score, 2),
                )

            except grpc.RpcError as exc:
                print("[WARN] AnalyzeFrame failed:", exc.code(), exc.details())

            last_sent = time.time()

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if not scores:
        return [], 0.0, "no_score"

    avg_score = sum(item["score"] for item in scores) / len(scores)
    main_label = max(set(labels), key=labels.count) if labels else "unknown"

    return scores, avg_score, main_label


def record_audio(duration_sec, save_path):
    print(f"[INFO] Recording answer for {duration_sec} seconds...")
    print("[INFO] Speak now.")

    audio = sd.rec(
        int(duration_sec * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()

    sf.write(str(save_path), audio, SAMPLE_RATE)
    print(f"[INFO] Saved answer audio: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default=SERVER_ADDRESS)
    parser.add_argument("--session-id", default="full-interview-session")
    parser.add_argument("--user-id", default="full-interview-user")
    parser.add_argument("--category", default="AI 백엔드 개발자")
    parser.add_argument("--interview-type", default="직무면접")
    parser.add_argument("--difficulty", default="normal")
    parser.add_argument("--question-count", type=int, default=2)
    parser.add_argument("--answer-duration", type=int, default=10)
    parser.add_argument("--language", default="ko-KR")
    parser.add_argument("--resume-text", default="AI 서버, gRPC, LLM, 음성 분석 기능을 구현한 지원자입니다.")

    args = parser.parse_args()

    channel = grpc.insecure_channel(args.server)
    stub = interview_pb2_grpc.InterviewAIServiceStub(channel)

    health = stub.HealthCheck(interview_pb2.HealthRequest(), timeout=5)
    print("[HEALTH]", health.status)
    print("[MODEL]", health.model_path)

    initial_request = interview_pb2.InitialQuestionGenerateRequest(
        session_id=args.session_id,
        user_id=args.user_id,
        category=args.category,
        interview_type=args.interview_type,
        difficulty=args.difficulty,
        question_count=args.question_count,
        time_per_question=args.answer_duration,
        resume_text=args.resume_text,
        language=args.language,
    )

    initial_response = stub.GenerateInitialQuestions(initial_request, timeout=60)

    question_results = []
    timeline_scores = []
    history = []

    for question in initial_response.questions:
        print("\n" + "=" * 80)
        print(f"[QUESTION #{question.index}]")
        print(question.question)
        print("=" * 80)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            audio_path = Path(temp_file.name)

        try:
            record_audio(args.answer_duration, audio_path)
            audio_bytes = audio_path.read_bytes()
        finally:
            try:
                audio_path.unlink(missing_ok=True)
            except Exception:
                pass

        subtitle_response = stub.GenerateSubtitles(
            interview_pb2.SubtitleRequest(
                session_id=args.session_id,
                user_id=args.user_id,
                audio=audio_bytes,
                audio_mime_type="audio/wav",
                language_hint=args.language,
            ),
            timeout=90,
        )

        answer_text = " ".join(segment.text for segment in subtitle_response.segments).strip()
        if not answer_text:
            answer_text = subtitle_response.summary

        print("\n[TRANSCRIBED ANSWER]")
        print(answer_text)

        voice_response = stub.AnalyzeVoiceTone(
            interview_pb2.VoiceToneAnalysisRequest(
                session_id=args.session_id,
                user_id=args.user_id,
                audio=audio_bytes,
                audio_mime_type="audio/wav",
                language_hint=args.language,
            ),
            timeout=30,
        )

        expression_samples, expression_avg, expression_label = collect_expression_scores(
            stub=stub,
            duration_sec=max(3, min(args.answer_duration, 8)),
            session_id=args.session_id,
            user_id=args.user_id,
        )

        total_score = (
            float(voice_response.overall_stability_score) * 0.45
            + float(expression_avg) * 0.35
            + 75.0 * 0.20
        )

        for sample in expression_samples:
            timeline_scores.append(
                interview_pb2.ScoreSample(
                    timestamp=sample["timestamp"],
                    expression_score=sample["score"],
                    voice_score=float(voice_response.overall_stability_score),
                    total_score=total_score,
                    expression_label=sample["label"],
                    note=sample["feedback"],
                )
            )

        follow_up_response = stub.GenerateFollowUpQuestion(
            interview_pb2.FollowUpQuestionGenerateRequest(
                session_id=args.session_id,
                user_id=args.user_id,
                previous_question=question.question,
                answer=answer_text,
                category=args.category,
                interview_type=args.interview_type,
                difficulty=args.difficulty,
                resume_text=args.resume_text,
                history=history,
                language=args.language,
            ),
            timeout=60,
        )

        follow_up_question = follow_up_response.question.question

        print("\n[FOLLOW-UP QUESTION]")
        print(follow_up_question)

        question_results.append(
            interview_pb2.QuestionAnalysisResult(
                index=question.index,
                question=question.question,
                answer=answer_text,
                follow_up_question=follow_up_question,
                subtitle_summary=subtitle_response.summary,
                voice_score=float(voice_response.overall_stability_score),
                expression_score=float(expression_avg),
                total_score=float(total_score),
                voice_feedback=voice_response.feedback,
                expression_feedback=f"주요 표정 라벨: {expression_label}",
            )
        )

        history.append(
            interview_pb2.InterviewTurn(
                index=question.index,
                question=question.question,
                answer=answer_text,
            )
        )

    report_response = stub.GenerateFinalReport(
        interview_pb2.FinalReportRequest(
            session_id=args.session_id,
            user_id=args.user_id,
            category=args.category,
            interview_type=args.interview_type,
            difficulty=args.difficulty,
            resume_text=args.resume_text,
            language=args.language,
            question_results=question_results,
            timeline_scores=timeline_scores,
        ),
        timeout=90,
    )

    print("\n" + "=" * 80)
    print("[FINAL REPORT]")
    print("=" * 80)
    print("overall_score:", report_response.overall_score)
    print("content_score:", report_response.content_score)
    print("voice_score:", report_response.voice_score)
    print("expression_score:", report_response.expression_score)
    print("\nsummary:")
    print(report_response.overall_summary)

    print("\nstrengths:")
    for item in report_response.strengths:
        print(f"- {item.title}: {item.detail}")

    print("\nweaknesses:")
    for item in report_response.weaknesses:
        print(f"- {item.title}: {item.detail} / improvement: {item.improvement}")

    print("\ntime-based insights:")
    for item in report_response.time_based_insights:
        print(f"- {item.time_range}: {item.observation} / {item.suggestion}")

    print("\nfinal advice:")
    print(report_response.final_advice)


if __name__ == "__main__":
    main()
