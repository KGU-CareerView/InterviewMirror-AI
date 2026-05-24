# InterviewMirror AI

InterviewMirror AI gRPC server.

이 서버는 클라이언트가 전달한 feature tensor를 ONNX 모델로 추론하고, Gemini API가 설정되어 있으면 면접 질문 생성, 후속 질문 생성, 자막 생성, 리포트 생성 기능도 함께 제공합니다.

## 요구 사항

- Docker
- gRPC 포트: `50051`
- ONNX 모델 파일: `models/interview_model_v3.onnx`
- 선택 환경 변수: `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_REPORT_MODEL`

## Docker 실행

이미지를 빌드합니다.

```bash
docker build -t interviewmirror-ai .
```

`.env` 파일이 있는 경우 다음처럼 실행합니다.

```bash
docker run --rm \
  --env-file .env \
  -p 50051:50051 \
  interviewmirror-ai
```

`.env` 파일이 없으면 다음처럼 실행할 수 있습니다.

```bash
docker run --rm \
  -p 50051:50051 \
  interviewmirror-ai
```

`GEMINI_API_KEY`가 없으면 Gemini 기반 기능은 비활성화되지만, feature tensor 기반 ONNX 추론 gRPC 서버는 실행됩니다.

## Docker 이미지에서 처리되는 작업

Dockerfile은 빌드 중 다음 작업을 수행합니다.

1. `requirements_grpc.txt` 의존성을 설치합니다.
2. 프로젝트 전체를 `/app`으로 복사합니다.
3. `proto/interview.proto`를 컴파일해 `generated` 모듈을 생성합니다.
4. 현재 프로젝트를 editable package로 설치합니다.
5. `python -m app.grpc_server`로 gRPC 서버를 실행합니다.

현재 프로젝트 설치 단계는 다음 명령으로 처리됩니다.

```bash
pip install --no-cache-dir --no-deps --no-build-isolation -e .
```

패키지 메타데이터는 `pyproject.toml`에 정의되어 있으며, 설치 대상 모듈은 `app*`, `generated*`입니다.

## 환경 변수

`.env` 예시:

```env
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.0-flash
GEMINI_REPORT_MODEL=gemini-3.1-pro-preview
```

`GEMINI_MODEL`은 질문 생성에 사용합니다. `GEMINI_REPORT_MODEL`은 최종 리포트 생성에 사용하며,
생략하면 기본값으로 `gemini-3.1-pro-preview`를 사용합니다.

## 로컬 실행

Docker 없이 로컬에서 실행할 수도 있습니다.

```bash
python -m pip install -r requirements_grpc.txt
python -m grpc_tools.protoc \
  -I proto \
  --python_out=generated \
  --grpc_python_out=generated \
  proto/interview.proto
python -c "from pathlib import Path; p=Path('generated/interview_pb2_grpc.py'); s=p.read_text(); s=s.replace('import interview_pb2 as interview__pb2', 'from generated import interview_pb2 as interview__pb2'); p.write_text(s); Path('generated/__init__.py').touch()"
python -m pip install --no-deps --no-build-isolation -e .
python -m app.grpc_server
```

서버가 정상 실행되면 `50051` 포트에서 gRPC 요청을 받습니다.

## 테스트 클라이언트

서버 실행 후 다른 터미널에서 테스트 클라이언트를 실행할 수 있습니다.

```bash
python tests/grpc_client_test.py
```

오디오 또는 전체 인터뷰 흐름 테스트가 필요하면 `tests/` 디렉터리의 다른 gRPC 클라이언트 스크립트를 사용할 수 있습니다.
