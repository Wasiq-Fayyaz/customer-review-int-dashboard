import json
import os

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "data", "outputs")
MODEL_PATH = os.path.join(BASE, "data", "sentiment_model.joblib")

app = FastAPI(title="Customer Review Intelligence API")

# Allows a separately hosted frontend (e.g. React dev server on :5173) to call this API
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://localhost:3000"],
                   allow_methods=["*"], allow_headers=["*"])

_bundle = None


def get_model():
    global _bundle
    if _bundle is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(503, "Model not trained yet. Run: python main.py pipeline")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


class ReviewIn(BaseModel):
    text: str


@app.get("/api/summary")
def summary():
    path = os.path.join(OUT_DIR, "dashboard_data.json")
    if not os.path.exists(path):
        raise HTTPException(503, "Dashboard data missing. Run: python main.py pipeline")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.post("/api/predict")
def predict(review: ReviewIn):
    text = review.text.strip()
    if not text:
        raise HTTPException(400, "Review text is empty")
    bundle = get_model()
    vec = bundle["vectorizer"].transform([text.lower()])
    model = bundle["model"]
    label = str(model.predict(vec)[0])
    # LinearSVC has no probabilities, so the signed distance is returned as a confidence signal
    score = float(model.decision_function(vec)[0]) if hasattr(model, "decision_function") else None
    return {"sentiment": label, "score": score, "model": bundle["name"]}


@app.get("/api/report")
def report():
    path = os.path.join(OUT_DIR, "customer_review_report.pdf")
    if not os.path.exists(path):
        raise HTTPException(404, "Report not generated. Run: python report.py")
    return FileResponse(path, filename="customer_review_report.pdf")


# Chart images (/outputs/wordcloud_positive.png, ...) and the dashboard UI
app.mount("/outputs", StaticFiles(directory=OUT_DIR), name="outputs")
app.mount("/", StaticFiles(directory=os.path.join(BASE, "frontend"), html=True), name="frontend")
