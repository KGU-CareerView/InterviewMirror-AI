from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional


app = FastAPI(title="Emotion Backend Server")


class EmotionResult(BaseModel):
    emotion: str
    confidence: float
    timestamp: Optional[float] = None


latest_result = {
    "emotion": None,
    "confidence": None,
    "timestamp": None
}


@app.get("/")
def root():
    return {"message": "Emotion backend server is running."}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/latest")
def get_latest():
    return latest_result


@app.post("/result")
def receive_result(result: EmotionResult):
    latest_result["emotion"] = result.emotion
    latest_result["confidence"] = result.confidence
    latest_result["timestamp"] = result.timestamp

    print(
        f"[RESULT] emotion={result.emotion}, "
        f"confidence={result.confidence:.4f}, "
        f"timestamp={result.timestamp}"
    )

    return {"status": "received", "data": latest_result}
