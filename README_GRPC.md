# InterviewMirror-AI gRPC Server

## 1. 설치

```bash
pip install -r requirements_grpc.txt
```

## 2. proto 컴파일

```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. interview.proto
```

## 3. 서버 실행

```bash
python grpc_server.py
```

## 4. 클라이언트 테스트

다른 터미널에서 실행합니다.

```bash
python grpc_client_test.py
```

## 구조

- `interview.proto`: gRPC API 명세
- `grpc_server.py`: 이미지 bytes 수신, 얼굴 검출, ONNX 추론, 결과 반환
- `grpc_client_test.py`: 웹캠 프레임을 서버로 보내는 테스트 클라이언트
- `face_cropper.py`: MediaPipe 기반 얼굴 crop 코드
- `interview_model_v3.onnx`: 면접 상태 분류 모델
- `face_detector.task`: MediaPipe 얼굴 검출 모델
