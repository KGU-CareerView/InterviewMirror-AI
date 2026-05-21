import argparse
import tempfile
from pathlib import Path

import grpc
import sounddevice as sd
import soundfile as sf

from generated import interview_pb2
from generated import interview_pb2_grpc


SERVER_ADDRESS = "127.0.0.1:50051"
SAMPLE_RATE = 16000


def record_audio(duration_sec: int, output_path: Path) -> None:
    print(f"[INFO] Recording for {duration_sec} seconds...")
    print("[INFO] Speak now.")

    audio = sd.rec(
        int(duration_sec * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()

    sf.write(str(output_path), audio, SAMPLE_RATE)

    print(f"[INFO] Saved temporary audio: {output_path}")


def print_health(stub):
    health = stub.HealthCheck(interview_pb2.HealthRequest(), timeout=5)

    print("[HEALTH]", health.status)
    print("[MODEL]", health.model_path)
    print("[QUESTION CLIENT]", health.question_client_status)
    print("[SUBTITLE CLIENT]", health.subtitle_client_status)
    print("[VOICE TONE ANALYZER]", health.voice_tone_analyzer_status)


def call_generate_subtitles(stub, audio_bytes: bytes, args):
    request = interview_pb2.SubtitleRequest(
        session_id=args.session_id,
        user_id=args.user_id,
        audio=audio_bytes,
        audio_mime_type="audio/wav",
        language_hint=args.language,
    )

    print("\n[INFO] Calling GenerateSubtitles...")

    try:
        response = stub.GenerateSubtitles(request, timeout=90)

        print("\n[SUBTITLE RESULT]")
        print("language:", response.language)
        print("summary:", response.summary)
        print("segments:", len(response.segments))

        for segment in response.segments:
            print(
                f"- #{segment.index} "
                f"{segment.start_ms}ms ~ {segment.end_ms}ms: "
                f"{segment.text}"
            )

        print("\n[SRT]")
        print(response.srt)

    except grpc.RpcError as exc:
        print("\n[SUBTITLE ERROR]")
        print(exc.code(), exc.details())


def call_analyze_voice_tone(stub, audio_bytes: bytes, args):
    request = interview_pb2.VoiceToneAnalysisRequest(
        session_id=args.session_id,
        user_id=args.user_id,
        audio=audio_bytes,
        audio_mime_type="audio/wav",
        language_hint=args.language,
    )

    print("\n[INFO] Calling AnalyzeVoiceTone...")

    try:
        response = stub.AnalyzeVoiceTone(request, timeout=30)

        print("\n[VOICE TONE RESULT]")
        print("pitch_mean:", response.pitch_mean)
        print("pitch_std:", response.pitch_std)
        print("pitch_stability:", response.pitch_stability)

        print("energy_mean:", response.energy_mean)
        print("energy_std:", response.energy_std)
        print("energy_stability:", response.energy_stability)

        print("pause_ratio:", response.pause_ratio)
        print("speech_duration_sec:", response.speech_duration_sec)
        print("total_duration_sec:", response.total_duration_sec)

        print("overall_stability_score:", response.overall_stability_score)
        print("feedback:", response.feedback)

    except grpc.RpcError as exc:
        print("\n[VOICE TONE ERROR]")
        print(exc.code(), exc.details())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=7)
    parser.add_argument("--server", default=SERVER_ADDRESS)
    parser.add_argument("--session-id", default="mic-test-session")
    parser.add_argument("--user-id", default="mic-test-user")
    parser.add_argument("--language", default="ko-KR")
    parser.add_argument("--save", default="", help="Optional path to save recorded wav")

    args = parser.parse_args()

    channel = grpc.insecure_channel(args.server)
    stub = interview_pb2_grpc.InterviewAIServiceStub(channel)

    print_health(stub)

    if args.save:
        audio_path = Path(args.save)
        record_audio(args.duration, audio_path)
        audio_bytes = audio_path.read_bytes()
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            audio_path = Path(temp_file.name)

        try:
            record_audio(args.duration, audio_path)
            audio_bytes = audio_path.read_bytes()
        finally:
            try:
                audio_path.unlink(missing_ok=True)
            except Exception:
                pass

    call_generate_subtitles(stub, audio_bytes, args)
    call_analyze_voice_tone(stub, audio_bytes, args)


if __name__ == "__main__":
    main()
