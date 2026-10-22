"""Task 5: FastAPI model-serving application."""
from pathlib import Path
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model.pkl"
if not MODEL_PATH.exists():
    raise FileNotFoundError("model.pkl missing. Run `python train_model.py` before starting the API.")
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Titanic Survival Prediction API",
    description="Predicts Titanic passenger survival using the saved scikit-learn pipeline.",
    version="1.0.0",
)

class PassengerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    Pclass: int = Field(..., ge=1, le=3, description="Ticket class: 1, 2, or 3")
    Sex: Literal["male", "female"]
    Age: float | None = Field(default=None, ge=0, le=100)
    SibSp: int = Field(default=0, ge=0, le=20)
    Parch: int = Field(default=0, ge=0, le=20)
    Fare: float = Field(..., ge=0)
    Embarked: Literal["C", "Q", "S"] | None = "S"

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True}

@app.post("/predict")
def predict(passenger: PassengerInput):
    # Use exactly the same feature names/order expected by the saved pipeline.
    record = pd.DataFrame([passenger.model_dump()])[
        ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]
    ]
    prediction = int(model.predict(record)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = round(float(model.predict_proba(record)[0][1]), 4)
    return {
        "prediction": prediction,
        "label": "Survived" if prediction == 1 else "Did not survive",
        "survival_probability": probability,
    }
