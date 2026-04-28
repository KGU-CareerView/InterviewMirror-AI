# InterviewMirror-AI gRPC Feature Server

## 변경된 구조

기존 구조는 다음과 같았습니다.

```text
client -> image bytes 전송
server -> MediaPipe FaceCropper + OpenCV 전처리 -> ONNX 추론
```

변경된 구조는 다음과 같습니다.

```text
client/front -> MediaPipe FaceCropper + OpenCV 전처리
client/front -> feature tensor 전송
server -> feature tensor를 받아 ONNX 추론만 수행
```

즉, 서버는 더 이상 이미지 bytes를 받지 않고 MediaPipe/OpenCV 전처리도 수행하지 않습니다.

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

## 파일 역할

- `interview.proto`: 이미지 bytes 대신 `FeatureRequest`를 사용하는 gRPC API 명세
- `grpc_client_test.py`: 웹캠 프레임에서 얼굴 검출, crop, OpenCV 전처리 후 feature tensor를 서버로 전송
- `grpc_server.py`: 전달받은 feature tensor를 ONNX 모델에 넣고 추론 결과 반환
- `face_cropper.py`: client/front 쪽에서 사용하는 MediaPipe 기반 얼굴 crop 코드
- `interview_model_v3.onnx`: 면접 상태 분류 모델
- `face_detector.task`: client/front 쪽 얼굴 검출 모델
