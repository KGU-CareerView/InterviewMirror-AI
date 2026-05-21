from dataclasses import dataclass
from pathlib import Path

import librosa
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
    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate

    def analyze(self, audio_path: Path) -> VoiceToneResult:
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        y, sr = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)

        if y.size == 0:
            return self._empty_result("음성 데이터가 비어 있습니다.")

        total_duration = float(librosa.get_duration(y=y, sr=sr))

        pitch_mean, pitch_std, pitch_stability = self._analyze_pitch(y, sr)
        energy_mean, energy_std, energy_stability = self._analyze_energy(y)
        pause_ratio, speech_duration = self._analyze_pause(y, sr, total_duration)

        pause_score = max(0.0, 100.0 - pause_ratio * 100.0)

        overall_score = (
            pitch_stability * 0.4
            + energy_stability * 0.35
            + pause_score * 0.25
        )

        feedback = self._make_feedback(
            overall_score=overall_score,
            pitch_stability=pitch_stability,
            energy_stability=energy_stability,
            pause_ratio=pause_ratio,
        )

        return VoiceToneResult(
            pitch_mean=round(pitch_mean, 2),
            pitch_std=round(pitch_std, 2),
            pitch_stability=round(pitch_stability, 2),

            energy_mean=round(energy_mean, 5),
            energy_std=round(energy_std, 5),
            energy_stability=round(energy_stability, 2),

            pause_ratio=round(pause_ratio, 3),
            speech_duration_sec=round(speech_duration, 2),
            total_duration_sec=round(total_duration, 2),

            overall_stability_score=round(overall_score, 2),
            feedback=feedback,
        )

    def _analyze_pitch(self, y: np.ndarray, sr: int) -> tuple[float, float, float]:
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

        pitch_values: list[float] = []

        for frame_idx in range(pitches.shape[1]):
            max_idx = int(np.argmax(magnitudes[:, frame_idx]))
            pitch = float(pitches[max_idx, frame_idx])

            if 50.0 <= pitch <= 500.0:
                pitch_values.append(pitch)

        if not pitch_values:
            return 0.0, 0.0, 50.0

        pitch_array = np.array(pitch_values, dtype=np.float32)
        pitch_mean = float(np.mean(pitch_array))
        pitch_std = float(np.std(pitch_array))

        pitch_stability = self._score_inverse_variation(
            value_std=pitch_std,
            reference=120.0,
        )

        return pitch_mean, pitch_std, pitch_stability

    def _analyze_energy(self, y: np.ndarray) -> tuple[float, float, float]:
        rms = librosa.feature.rms(y=y)[0]

        energy_mean = float(np.mean(rms))
        energy_std = float(np.std(rms))

        energy_stability = self._score_inverse_variation(
            value_std=energy_std,
            reference=0.05,
        )

        return energy_mean, energy_std, energy_stability

    def _analyze_pause(
        self,
        y: np.ndarray,
        sr: int,
        total_duration: float,
    ) -> tuple[float, float]:
        intervals = librosa.effects.split(y, top_db=30)

        speech_samples = 0

        for start, end in intervals:
            speech_samples += int(end - start)

        speech_duration = speech_samples / sr

        if total_duration <= 0:
            return 1.0, 0.0

        pause_ratio = max(0.0, 1.0 - speech_duration / total_duration)

        return pause_ratio, speech_duration

    def _score_inverse_variation(self, value_std: float, reference: float) -> float:
        if reference <= 0:
            return 0.0

        score = 100.0 - min(value_std / reference, 1.0) * 100.0

        return max(0.0, min(100.0, score))

    def _make_feedback(
        self,
        overall_score: float,
        pitch_stability: float,
        energy_stability: float,
        pause_ratio: float,
    ) -> str:
        feedbacks: list[str] = []

        if overall_score >= 80:
            feedbacks.append("전반적으로 목소리 톤이 안정적으로 유지되었습니다.")
        elif overall_score >= 60:
            feedbacks.append("대체로 안정적이지만 일부 구간에서 톤 변화가 감지되었습니다.")
        else:
            feedbacks.append("목소리 톤의 흔들림이 비교적 크게 나타났습니다.")

        if pitch_stability < 60:
            feedbacks.append("목소리 높낮이 변화가 커서 긴장감이 드러날 수 있습니다.")

        if energy_stability < 60:
            feedbacks.append("음량 변화가 불규칙하여 답변 전달력이 떨어질 수 있습니다.")

        if pause_ratio > 0.4:
            feedbacks.append("침묵 구간이 많아 답변 흐름이 끊겨 보일 수 있습니다.")

        return " ".join(feedbacks)

    def _empty_result(self, message: str) -> VoiceToneResult:
        return VoiceToneResult(
            pitch_mean=0.0,
            pitch_std=0.0,
            pitch_stability=0.0,

            energy_mean=0.0,
            energy_std=0.0,
            energy_stability=0.0,

            pause_ratio=1.0,
            speech_duration_sec=0.0,
            total_duration_sec=0.0,

            overall_stability_score=0.0,
            feedback=message,
        )
