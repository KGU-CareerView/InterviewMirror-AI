import argparse
from pathlib import Path

import grpc

from generated import interview_pb2
from generated import interview_pb2_grpc


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--server", default="127.0.0.1:50051")
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

    health = stub.HealthCheck(interview_pb2.HealthRequest(), timeout=5)

    print("[HEALTH]", health.status)
    print("[MODEL]", health.model_path)
    print("[QUESTION CLIENT]", health.question_client_status)
    print("[SUBTITLE CLIENT]", health.subtitle_client_status)
    print("[VOICE TONE ANALYZER]", health.voice_tone_analyzer_status)

    subtitle_request = interview_pb2.SubtitleRequest(
        session_id=args.session_id,
        user_id=args.user_id,
        audio=audio_bytes,
        audio_mime_type=audio_mime_type,
        language_hint=args.language,
    )

    try:
        subtitle_response = stub.GenerateSubtitles(subtitle_request, timeout=60)

        print("\n[SUBTITLE]")
        print("language:", subtitle_response.language)
        print("summary:", subtitle_response.summary)
        print("segments:", len(subtitle_response.segments))
        print("srt:")
        print(subtitle_response.srt)

    except grpc.RpcError as exc:
        print("\n[SUBTITLE ERROR]")
        print(exc.code(), exc.details())

    tone_request = interview_pb2.VoiceToneAnalysisRequest(
        session_id=args.session_id,
        user_id=args.user_id,
        audio=audio_bytes,
        audio_mime_type=audio_mime_type,
        language_hint=args.language,
    )

    try:
        tone_response = stub.AnalyzeVoiceTone(tone_request, timeout=30)

        print("\n[VOICE TONE]")
        print("pitch_mean:", tone_response.pitch_mean)
        print("pitch_std:", tone_response.pitch_std)
        print("pitch_stability:", tone_response.pitch_stability)
        print("energy_mean:", tone_response.energy_mean)
        print("energy_std:", tone_response.energy_std)
        print("energy_stability:", tone_response.energy_stability)
        print("pause_ratio:", tone_response.pause_ratio)
        print("speech_duration_sec:", tone_response.speech_duration_sec)
        print("total_duration_sec:", tone_response.total_duration_sec)
        print("overall_stability_score:", tone_response.overall_stability_score)
        print("feedback:", tone_response.feedback)

    except grpc.RpcError as exc:
        print("\n[VOICE TONE ERROR]")
        print(exc.code(), exc.details())


if __name__ == "__main__":
    main()
