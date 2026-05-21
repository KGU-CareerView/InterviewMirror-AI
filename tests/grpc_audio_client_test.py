import argparse
from pathlib import Path

import grpc

from generated import interview_pb2
from generated import interview_pb2_grpc


SERVER_ADDRESS = "127.0.0.1:50051"


def guess_mime_type(audio_path: Path) -> str:
    suffix = audio_path.suffix.lower()

    if suffix == ".wav":
        return "audio/wav"

    if suffix == ".mp3":
        return "audio/mpeg"

    if suffix == ".webm":
        return "audio/webm"

    if suffix == ".ogg":
        return "audio/ogg"

    if suffix in [".m4a", ".mp4"]:
        return "audio/mp4"

    return "audio/wav"


def print_health(stub):
    health = stub.HealthCheck(interview_pb2.HealthRequest(), timeout=5)

    print("[HEALTH]", health.status)
    print("[MODEL]", health.model_path)
    print("[QUESTION CLIENT]", health.question_client_status)
    print("[SUBTITLE CLIENT]", health.subtitle_client_status)
    print("[VOICE TONE ANALYZER]", health.voice_tone_analyzer_status)


def test_subtitle(stub, audio_bytes, audio_mime_type, args):
    request = interview_pb2.SubtitleRequest(
        session_id=args.session_id,
        user_id=args.user_id,
        audio=audio_bytes,
        audio_mime_type=audio_mime_type,
        language_hint=args.language,
    )

    print("\n[INFO] Testing GenerateSubtitles...")

    try:
        response = stub.GenerateSubtitles(request, timeout=90)

        print("\n[SUBTITLE]")
        print("session_id:", response.session_id)
        print("user_id:", response.user_id)
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


def test_voice_tone(stub, audio_bytes, audio_mime_type, args):
    request = interview_pb2.VoiceToneAnalysisRequest(
        session_id=args.session_id,
        user_id=args.user_id,
        audio=audio_bytes,
        audio_mime_type=audio_mime_type,
        language_hint=args.language,
    )

    print("\n[INFO] Testing AnalyzeVoiceTone...")

    try:
        response = stub.AnalyzeVoiceTone(request, timeout=30)

        print("\n[VOICE TONE]")
        print("session_id:", response.session_id)
        print("user_id:", response.user_id)

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
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--server", default=SERVER_ADDRESS)
    parser.add_argument("--session-id", default="audio-test-session")
    parser.add_argument("--user-id", default="audio-test-user")
    parser.add_argument("--language", default="ko-KR")

    args = parser.parse_args()

    audio_path = Path(args.audio)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio_bytes = audio_path.read_bytes()
    audio_mime_type = guess_mime_type(audio_path)

    channel = grpc.insecure_channel(args.server)
    stub = interview_pb2_grpc.InterviewAIServiceStub(channel)

    print_health(stub)

    test_subtitle(
        stub=stub,
        audio_bytes=audio_bytes,
        audio_mime_type=audio_mime_type,
        args=args,
    )

    test_voice_tone(
        stub=stub,
        audio_bytes=audio_bytes,
        audio_mime_type=audio_mime_type,
        args=args,
    )


if __name__ == "__main__":
    main()
