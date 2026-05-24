# AI 서버 음성 분석 설계 변경 명세

## 배경

기존 `interview.proto`의 `VoiceToneAnalysisRequest`는 `bytes audio`로 오디오 파일을 직접 수신하는 구조였으나,
VOICE.md 설계 원칙("raw PCM/오디오 파일은 전송하지 않습니다")에 따라 **사전 집계된 특징값**을 수신하는 방식으로 변경합니다.

---

## 1. proto 변경 사항 요약

- `GenerateSubtitles` RPC 및 관련 메시지 삭제 (STT는 프론트 Web Speech API 담당)
- `VoiceToneAnalysisRequest`: `bytes audio` → `AudioSummaryData` + `repeated float zcr_samples`
- `QuestionAnalysisResult`: `audio_summary`, `zcr_samples` 필드 추가 (GenerateFinalReport 경로)
- `HealthResponse.subtitle_client_status` 필드 삭제

---

## 2. 데이터 흐름 (변경 후)

```
[realtime.audio 수신 시 — 1초마다]
ZCR → Redis session:{id}:zcr:{questionIndex} 리스트 적재

[session.answer 수신 시 — 질문당 1회]
audioSummary → InterviewDetail.audioSummaryJson 저장
               AudioScoreService.calculateQuestionScore() → InterviewDetail.audioScore 저장

[realtime.end 수신 시]
FinalReportService.requestFinalReport(sessionId)
  ├── InterviewDetail 목록 로드 (question, answer, audioSummaryJson, audioScore)
  ├── Redis에서 질문별 ZCR 샘플 조회
  └── GenerateFinalReport gRPC (async) ──▶ AI 서버
                                            └── saveReportFromGrpc()
```

---

## 3. Python `VoiceToneAnalyzer` 구현

아래는 `audio_path` 기반 코드를 `FinalReportRequest.QuestionAnalysisResult`에서 받은 집계 특징값 기반으로 전면 재작성한 구현입니다.

### 3-1. 데이터 클래스 (변경 없음)

```python
@dataclass
class VoiceToneResult:
    pitch_mean: float          # ZCR mean (0~1 스케일, ZCR 없으면 0.0)
    pitch_std: float           # ZCR std  (ZCR 없으면 0.0)
    pitch_stability: float     # ZCR 기반 안정성 점수 0~100 (없으면 50.0)

    energy_mean: float         # avg_rms
    energy_std: float          # avg_rms * rms_cov
    energy_stability: float    # max(0, 100 * (1 - rms_cov / 0.5))

    pause_ratio: float         # 1 - speech_ratio
    speech_duration_sec: float # speech_ratio * response_time_sec
    total_duration_sec: float  # response_time_sec

    overall_stability_score: float
    feedback: str
```

### 3-2. `VoiceToneAnalyzer` 전체 구현

```python
from dataclasses import dataclass
import numpy as np


class VoiceToneAnalyzer:

    def analyze_from_features(
        self,
        audio_summary,          # AudioSummaryData proto message
        zcr_samples: list[float],
        response_time_seconds: int,
    ) -> VoiceToneResult:
        """
        audio_summary: proto AudioSummaryData
        zcr_samples:   초당 ZCR 값 목록 (비어있을 수 있음)
        response_time_seconds: 전체 응답 시간(초)
        """
        response_time_sec = float(response_time_seconds) if response_time_seconds > 0 else 1.0

        # --- 에너지 분석 ---
        avg_rms = audio_summary.avg_rms or 0.0
        rms_cov = audio_summary.rms_cov or 0.0
        energy_mean = avg_rms
        energy_std = avg_rms * rms_cov
        energy_stability = max(0.0, 100.0 * (1.0 - rms_cov / 0.5))

        # --- 포즈/침묵 분석 ---
        speech_ratio = audio_summary.speech_ratio or 0.0
        pause_ratio = max(0.0, 1.0 - speech_ratio)
        speech_duration_sec = speech_ratio * response_time_sec
        total_duration_sec = response_time_sec

        # --- 피치 분석 (ZCR 기반) ---
        zcr_provided = len(zcr_samples) > 0
        if zcr_provided:
            pitch_mean, pitch_std, pitch_stability = self._analyze_pitch_from_zcr(zcr_samples)
        else:
            pitch_mean, pitch_std, pitch_stability = 0.0, 0.0, 50.0  # neutral

        # --- 종합 점수 ---
        pause_score = max(0.0, 100.0 - pause_ratio * 100.0)
        if zcr_provided:
            overall_score = (
                pitch_stability * 0.4
                + energy_stability * 0.35
                + pause_score * 0.25
            )
        else:
            overall_score = energy_stability * 0.6 + pause_score * 0.4

        feedback = self._make_feedback(
            overall_score=overall_score,
            pitch_stability=pitch_stability,
            energy_stability=energy_stability,
            pause_ratio=pause_ratio,
            zcr_provided=zcr_provided,
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

        # ZCR std 기준 reference=0.05: std가 0.05 이상이면 불안정으로 판정
        pitch_stability = self._score_inverse_variation(
            value_std=zcr_std, reference=0.05
        )
        return zcr_mean, zcr_std, pitch_stability

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
        zcr_provided: bool,
    ) -> str:
        feedbacks: list[str] = []

        if overall_score >= 80:
            feedbacks.append("전반적으로 목소리 톤이 안정적으로 유지되었습니다.")
        elif overall_score >= 60:
            feedbacks.append("대체로 안정적이지만 일부 구간에서 톤 변화가 감지되었습니다.")
        else:
            feedbacks.append("목소리 톤의 흔들림이 비교적 크게 나타났습니다.")

        if zcr_provided and pitch_stability < 60:
            feedbacks.append("목소리 높낮이 변화가 커서 긴장감이 드러날 수 있습니다.")

        if energy_stability < 60:
            feedbacks.append("음량 변화가 불규칙하여 답변 전달력이 떨어질 수 있습니다.")

        if pause_ratio > 0.4:
            feedbacks.append("침묵 구간이 많아 답변 흐름이 끊겨 보일 수 있습니다.")

        return " ".join(feedbacks)
```

### 3-3. gRPC 핸들러 연결

`GenerateFinalReport` 핸들러 내부에서 질문별로 `analyze_from_features()`를 호출합니다.

```python
class InterviewAIServiceServicer(InterviewAIServiceServicer):

    def __init__(self):
        self.voice_analyzer = VoiceToneAnalyzer()

    def GenerateFinalReport(self, request, context):
        question_results = []
        for qa in request.question_results:
            voice_result = self.voice_analyzer.analyze_from_features(
                audio_summary=qa.audio_summary,
                zcr_samples=list(qa.zcr_samples),
                response_time_seconds=qa.response_time_seconds if hasattr(qa, 'response_time_seconds') else 60,
            )
            question_results.append({
                "index": qa.index,
                "voice_stability_score": voice_result.overall_stability_score,
                "voice_feedback": voice_result.feedback,
                "pitch_stability": voice_result.pitch_stability,
                "energy_stability": voice_result.energy_stability,
                "pause_ratio": voice_result.pause_ratio,
            })

        # 기존 LLM 기반 리포트 생성 로직에 voice_result 통합
        # ... (existing report generation)

    def AnalyzeVoiceTone(self, request, context):
        result = self.voice_analyzer.analyze_from_features(
            audio_summary=request.audio_summary,
            zcr_samples=list(request.zcr_samples),
            response_time_seconds=request.response_time_seconds,
        )
        return VoiceToneAnalysisResponse(
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
```

### 3-4. 삭제 대상

- `VoiceToneAnalyzer.analyze(audio_path)` 기존 메서드
- `_analyze_pitch()` (librosa.piptrack 기반)
- `_analyze_energy()` (librosa.feature.rms 기반)
- `_analyze_pause()` (librosa.effects.split 기반)
- `_empty_result()` (audio 파일 없을 때 fallback, 더 이상 불필요)
- `GenerateSubtitles` 핸들러 및 관련 STT 로직

---

## 4. `VoiceToneResult` 필드 출처 매핑

| 필드 | 기존 (librosa) | 변경 후 |
|---|---|---|
| `pitch_mean` | piptrack Hz 평균 | ZCR mean (0~1 스케일, 없으면 0.0) |
| `pitch_std` | piptrack Hz 표준편차 | ZCR std (없으면 0.0) |
| `pitch_stability` | piptrack 기반 0~100 | ZCR std 역변환 0~100 (없으면 50.0) |
| `energy_mean` | librosa RMS mean | `audio_summary.avg_rms` |
| `energy_std` | librosa RMS std | `avg_rms * rms_cov` |
| `energy_stability` | RMS std 역변환 | `max(0, 100*(1 - rms_cov/0.5))` |
| `pause_ratio` | librosa effects.split | `1 - speech_ratio` |
| `speech_duration_sec` | librosa | `speech_ratio * response_time_sec` |
| `total_duration_sec` | librosa | `response_time_sec` |
| `overall_stability_score` | pitch·energy·pause 가중합 | ZCR 있을 때 동일, 없을 때 가중치 재조정 |
| `feedback` | rule-based | 동일 (pitch 항목은 ZCR 있을 때만) |

---

## 5. 의존성 변경

| 항목 | 변경 |
|---|---|
| `librosa` | **제거 가능** — 음성 파일 처리가 없어져 불필요 |
| `numpy` | 유지 (ZCR 통계 계산에 사용) |
| `pathlib.Path` | **제거** — 파일 경로 불필요 |

---

## 6. `GenerateFinalReport` 입출력 확장

최종 보고서 생성 시 AI 서버가 처리해야 할 정보가 추가되었습니다.

### 6-1. `FinalReportRequest` 입력 필드 (백엔드 → AI)

| 필드 | 타입 | 설명 |
|---|---|---|
| `session_id`, `user_id` | string | 식별자 |
| `category` | string | 면접 분야 (예: BACKEND, FRONTEND) |
| `interview_type` | string | 인성/직무/종합 |
| `difficulty` | string | 난이도 |
| `resume_text` | string | 자소서 내용 (답변 정확도 채점 컨텍스트) |
| `language` | string | "ko" |
| `question_results` | repeated `QuestionAnalysisResult` | 질문별 데이터 (아래) |
| `emotion_graph_json` | string | **신규** — 세션 전체 표정 감정 변화 타임라인 JSON |

### 6-2. `QuestionAnalysisResult` 필드 (질문별)

| 필드 | 타입 | 설명 |
|---|---|---|
| `index` | int32 | 질문 번호 (1-based) |
| `question` | string | 질문 텍스트 |
| `answer` | string | **STT 변환된 답변 (대본)** |
| `answer_length` | int32 | **신규** — 답변 글자 수 |
| `response_time_seconds` | int32 | **신규** — 답변 소요 시간 (초) |
| `emotion_result_json` | string | **신규** — 질문별 표정 감정 분석 결과 JSON |
| `voice_score` | double | 백엔드가 `AnalyzeVoiceTone`으로 미리 산출한 음성 점수 (0~100, AI는 참고만) |
| `audio_summary` | `AudioSummaryData` | 음성 분석 raw 데이터 (rms, wpm, pause, ttr 등) |
| `zcr_samples` | repeated float | ZCR 시계열 (1초마다 1샘플) |

### 6-3. `FinalReportResponse` 출력 필드 (AI → 백엔드)

기존 필드 유지 + **신규 추가**:

| 필드 | 타입 | 설명 |
|---|---|---|
| `overall_score` | double | 종합 점수 (0~100) — 아래 4가지 종합 |
| `content_score` | double | 답변 내용/정확도 점수 |
| `voice_score` | double | 목소리 점수 |
| `expression_score` | double | 표정 점수 |
| `overall_summary` | string | 전체 요약 |
| `final_advice` | string | 종합 조언 |
| `strengths` | repeated `ReportStrengthMessage` | 강점 목록 (`title`, `detail`) |
| `weaknesses` | repeated `ReportWeaknessMessage` | 보완점 (`title`, `detail`, `improvement`) |
| `time_based_insights` | repeated `ReportTimeInsightMessage` | 시간대별 인사이트 |
| `question_feedbacks` | repeated `QuestionFeedback` | **신규 — 질문별 피드백** |

### 6-4. `QuestionFeedback` 메시지 (신규)

```protobuf
message QuestionFeedback {
  int32 index = 1;                  // 질문 번호 (request의 index와 매칭)
  double total_score = 2;           // 질문별 종합 점수
  double content_score = 3;         // 답변 내용/정확도 점수
  double voice_score = 4;           // 목소리 점수
  double expression_score = 5;      // 표정 점수
  string overall_feedback = 6;      // 질문별 종합 피드백
  string content_feedback = 7;      // 답변 내용 피드백
  string voice_feedback = 8;        // 목소리 피드백
  string expression_feedback = 9;   // 표정 피드백
}
```

### 6-5. 최종 점수 산출 가이드

`overall_score`는 아래 4가지를 종합한 점수입니다.

1. **표정 감정 분석** — `emotion_graph_json` (세션 전체) + `question_results[].emotion_result_json` (질문별)
2. **답변 정확도/내용** — `question_results[].answer` (STT 대본) + `category` + `resume_text`로 평가
3. **답변 길이/응답 시간** — `question_results[].answer_length` + `question_results[].response_time_seconds`
4. **목소리 분석** — `question_results[].audio_summary` (raw 특징값) + `zcr_samples` + 사전 계산된 `voice_score`

질문별 점수도 동일 4가지 기준을 적용하여 `QuestionFeedback`에 채워서 응답해야 합니다.
