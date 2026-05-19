from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "interview_model_v3.onnx"
MEDIAPIPE_FACE_TASK_MODEL = BASE_DIR / "models" / "face_detector.task"

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

INPUT_SIZE = 224

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

MIN_DETECTION_CONFIDENCE = 0.6
BBOX_MARGIN = 0.25

USE_BACKEND = False
BACKEND_RESULT_URL = "http://127.0.0.1:8000/result"

SEND_EVERY_N_FRAMES = 10
DISPLAY_CONFIDENCE_THRESHOLD = 0.0
