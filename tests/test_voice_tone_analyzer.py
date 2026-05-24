from types import SimpleNamespace

from app.voice.voice_tone_analyzer import VoiceToneAnalyzer


def _audio_summary(speech_ratio: float):
    return SimpleNamespace(
        avg_rms=0.02,
        rms_cov=0.0,
        speech_ratio=speech_ratio,
        word_count=10,
    )


def test_high_pause_ratio_caps_voice_score():
    analyzer = VoiceToneAnalyzer()

    result = analyzer.analyze_from_features(
        audio_summary=_audio_summary(speech_ratio=0.19),
        zcr_samples=[0.02] * 20,
        response_time_seconds=100,
    )

    assert result.pause_ratio == 0.81
    assert result.overall_stability_score == 55.0
    assert "크게 감점" in result.feedback


def test_low_pause_ratio_can_still_score_high():
    analyzer = VoiceToneAnalyzer()

    result = analyzer.analyze_from_features(
        audio_summary=_audio_summary(speech_ratio=0.80),
        zcr_samples=[0.02] * 20,
        response_time_seconds=100,
    )

    assert result.pause_ratio == 0.2
    assert result.overall_stability_score == 91.0


def test_insufficient_speech_caps_voice_score_as_non_answer():
    analyzer = VoiceToneAnalyzer()

    result = analyzer.analyze_from_features(
        audio_summary=SimpleNamespace(
            avg_rms=0.02,
            rms_cov=0.0,
            speech_ratio=0.10,
            word_count=0,
        ),
        zcr_samples=[0.02] * 20,
        response_time_seconds=100,
    )

    assert result.pause_ratio == 0.9
    assert result.overall_stability_score == 20.0
    assert "무응답에 가깝게 처리" in result.feedback


def test_placeholder_transcript_caps_voice_score_even_with_detected_speech():
    analyzer = VoiceToneAnalyzer()

    result = analyzer.analyze_from_features(
        audio_summary=SimpleNamespace(
            avg_rms=0.02,
            rms_cov=0.334,
            speech_ratio=0.39,
            word_count=3,
        ),
        zcr_samples=[0.02, 0.07, 0.01, 0.04, 0.08],
        response_time_seconds=7,
        transcript_is_placeholder=True,
    )

    assert result.pause_ratio == 0.61
    assert result.overall_stability_score == 5.0
    assert "STT 기본 문구만 감지" in result.feedback
