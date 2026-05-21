def __init__(self):
    print(f"[INFO] Loading ONNX model: {MODEL_PATH}")
    self.session = ort.InferenceSession(str(MODEL_PATH))
    self.input_name = self.session.get_inputs()[0].name

    self.question_client = self._init_question_client()
    self.subtitle_client = self._init_subtitle_client()
    self.voice_tone_analyzer = self._init_voice_tone_analyzer()