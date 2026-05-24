from dataclasses import dataclass

import numpy as np


@dataclass
class VoiceToneResult:
    pitch_mean: float
    pitch_std: float
    pitch_stability: float

    energy_mean: float
    energy_std: float
    energy_stability: float

    pause_ratio: float
    speech_duration_sec: float
    total_duration_sec: float

    overall_stability_score: float
    feedback: str


class VoiceToneAnalyzer:

    def analyze_from_features(
        self,
        audio_summary,
        zcr_samples: list[float],
        response_time_seconds: int,
        transcript_is_placeholder: bool = False,
    ) -> VoiceToneResult:
        response_time_sec = float(response_time_seconds) if response_time_seconds > 0 else 1.0

        avg_rms = audio_summary.avg_rms or 0.0
        rms_cov = audio_summary.rms_cov or 0.0
        energy_mean = avg_rms
        energy_std = avg_rms * rms_cov
        energy_stability = max(0.0, 100.0 * (1.0 - rms_cov / 0.5))

        speech_ratio = audio_summary.speech_ratio or 0.0
        pause_ratio = max(0.0, 1.0 - speech_ratio)
        speech_duration_sec = speech_ratio * response_time_sec
        total_duration_sec = response_time_sec
        word_count = getattr(audio_summary, "word_count", None)
        has_meaningful_speech = self._has_meaningful_speech(
            speech_ratio=speech_ratio,
            word_count=word_count,
            transcript_is_placeholder=transcript_is_placeholder,
        )

        zcr_provided = len(zcr_samples) > 0
        if zcr_provided:
            pitch_mean, pitch_std, pitch_stability = self._analyze_pitch_from_zcr(zcr_samples)
        else:
            pitch_mean, pitch_std, pitch_stability = 0.0, 0.0, 50.0

        pause_score = max(0.0, 100.0 - pause_ratio * 100.0)
        if zcr_provided:
            overall_score = (
                pitch_stability * 0.30
                + energy_stability * 0.25
                + pause_score * 0.45
            )
        else:
            overall_score = energy_stability * 0.35 + pause_score * 0.65

        overall_score = min(
            overall_score,
            self._pause_score_cap(pause_ratio),
        )
        if transcript_is_placeholder:
            overall_score = 5.0
        elif not has_meaningful_speech:
            overall_score = min(overall_score, 20.0)

        feedback = self._make_feedback(
            overall_score=overall_score,
            pitch_stability=pitch_stability,
            energy_stability=energy_stability,
            pause_ratio=pause_ratio,
            zcr_provided=zcr_provided,
            has_meaningful_speech=has_meaningful_speech,
            transcript_is_placeholder=transcript_is_placeholder,
        )

        return VoiceToneResult(
            pitch_mean=round(pitch_mean, 4),
            pitch_std=round(pitch_std, 4),
            pitch_stability=round(pitch_stability, 2),
            energy_mean=round(energy_mean, 5),
            energy_std=round(energy_std, 5),
            energy_stability=round(energy_stability, 2),
            pause_ratio=round(pause_ratio, 3),
            speech_duration_sec=round(speech_duration_sec, 2),
            total_duration_sec=round(total_duration_sec, 2),
            overall_stability_score=round(overall_score, 2),
            feedback=feedback,
        )

    def _analyze_pitch_from_zcr(
        self, zcr_samples: list[float]
    ) -> tuple[float, float, float]:
        zcr_array = np.array(zcr_samples, dtype=np.float32)
        zcr_mean = float(np.mean(zcr_array))
        zcr_std = float(np.std(zcr_array))

        # ZCR std reference=0.05: std >= 0.05 → 불안정 판정
        pitch_stability = self._score_inverse_variation(value_std=zcr_std, reference=0.05)
        return zcr_mean, zcr_std, pitch_stability

    def _score_inverse_variation(self, value_std: float, reference: float) -> float:
        if reference <= 0:
            return 0.0
        score = 100.0 - min(value_std / reference, 1.0) * 100.0
        return max(0.0, min(100.0, score))

    def _pause_score_cap(self, pause_ratio: float) -> float:
        if pause_ratio >= 0.80:
            return 55.0
        if pause_ratio >= 0.70:
            return 60.0
        if pause_ratio >= 0.60:
            return 70.0
        if pause_ratio >= 0.50:
            return 80.0
        return 100.0

    def _has_meaningful_speech(
        self,
        speech_ratio: float,
        word_count: int | None,
        transcript_is_placeholder: bool,
    ) -> bool:
        if transcript_is_placeholder:
            return False
        if speech_ratio < 0.15:
            return False
        if word_count is not None and word_count <= 0 and speech_ratio < 0.30:
            return False
        return True

    def _make_feedback(
        self,
        overall_score: float,
        pitch_stability: float,
        energy_stability: float,
        pause_ratio: float,
        zcr_provided: bool,
        has_meaningful_speech: bool,
        transcript_is_placeholder: bool,
    ) -> str:
        feedbacks: list[str] = []

        if transcript_is_placeholder:
            feedbacks.append("STT 기본 문구만 감지되어 실제 답변이 없는 것으로 처리되었습니다.")
            feedbacks.append("음성 점수는 무응답 기준 최저점에 가깝게 고정되었습니다.")
            return " ".join(feedbacks)

        if not has_meaningful_speech:
            feedbacks.append("실제 발화가 충분하지 않아 음성 평가는 무응답에 가깝게 처리되었습니다.")
            if pause_ratio >= 0.7:
                feedbacks.append("침묵 비율이 매우 높아 답변 전달력 점수가 크게 감점되었습니다.")
            elif pause_ratio > 0.4:
                feedbacks.append("침묵 구간이 많아 답변 흐름이 끊겨 보일 수 있습니다.")
            return " ".join(feedbacks)
        elif overall_score >= 80:
            feedbacks.append("전반적으로 목소리 톤이 안정적으로 유지되었습니다.")
        elif overall_score >= 60:
            feedbacks.append("대체로 안정적이지만 일부 구간에서 톤 변화가 감지되었습니다.")
        else:
            feedbacks.append("목소리 톤의 흔들림이 비교적 크게 나타났습니다.")

        if zcr_provided and pitch_stability < 60:
            feedbacks.append("목소리 높낮이 변화가 커서 긴장감이 드러날 수 있습니다.")

        if energy_stability < 60:
            feedbacks.append("음량 변화가 불규칙하여 답변 전달력이 떨어질 수 있습니다.")

        if pause_ratio >= 0.7:
            feedbacks.append("침묵 비율이 매우 높아 답변 전달력 점수가 크게 감점되었습니다.")
        elif pause_ratio > 0.4:
            feedbacks.append("침묵 구간이 많아 답변 흐름이 끊겨 보일 수 있습니다.")

        return " ".join(feedbacks)
