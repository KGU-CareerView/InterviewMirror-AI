from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort
import time

# 모델 경로 설정
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "interview_model_v3.onnx"

# 클래스 매핑
CLASS_NAMES = {0: 'Stable', 1: 'Nervous', 2: 'Neutral'}
FEEDBACK = {
    'Stable': '표정이 안정적이고 자신감 있어 보입니다.',
    'Nervous': '긴장감이 다소 보입니다. 심호흡해보세요.',
    'Neutral': '표정이 중립적입니다. 자연스러운 미소를 더해보세요.'
}

# ONNX 세션 로드
sess = ort.InferenceSession(MODEL_PATH)
input_name = sess.get_inputs()[0].name

# Haar Cascade 얼굴 검출
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

def preprocess(face_bgr):
    face = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (224, 224))
    face = face.astype(np.float32) / 255.0
    face = (face - MEAN) / STD
    face = np.transpose(face, (2, 0, 1))
    face = np.expand_dims(face, axis=0).astype(np.float32)
    return face

def softmax(logits):
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / np.sum(exp)

# 웹캠 실행
cap = cv2.VideoCapture(0)
print("✅ 웹캠 시작! 'q' 누르면 종료")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # 여백 추가
        margin = int(0.2 * min(w, h))
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        # 전처리 + 추론
        tensor = preprocess(face_crop)
        outputs = sess.run(None, {input_name: tensor})
        probs = softmax(outputs[0][0])
        pred_idx = int(np.argmax(probs))
        label = CLASS_NAMES[pred_idx]
        confidence = float(probs[pred_idx])
        feedback = FEEDBACK[label]

        # 화면에 표시
        color = (0,255,0) if label=='Stable' else (0,0,255) if label=='Nervous' else (255,165,0)
        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        cv2.putText(frame, f'{label} {confidence:.2f}',
                    (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame, feedback,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # CMD 출력
        print(f'label={label}, confidence={confidence:.4f}, feedback={feedback}')

    cv2.imshow('Interview Emotion Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()