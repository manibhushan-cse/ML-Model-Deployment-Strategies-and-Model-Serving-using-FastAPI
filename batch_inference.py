"""Task 3: Predict 20 held-out passenger records together."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train.csv"
MODEL_PATH = ROOT / "model.pkl"
OUTPUT_PATH = ROOT / "batch_predictions.csv"
FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("model.pkl missing. Run: python train_model.py")
    df = pd.read_csv(DATA_PATH)
    _, X_test, _, _ = train_test_split(
        df[FEATURES], df["Survived"].astype(int),
        test_size=0.20, random_state=42, stratify=df["Survived"].astype(int)
    )
    if len(X_test) < 20:
        raise ValueError("At least 20 held-out records are required for batch inference.")
    batch = X_test.iloc[:20].copy()
    model = joblib.load(MODEL_PATH)
    results = batch.copy()
    results.insert(0, "PassengerId", df.loc[batch.index, "PassengerId"].values)
    results["PredictedSurvived"] = model.predict(batch).astype(int)
    results["PredictionLabel"] = results["PredictedSurvived"].map(
        {0: "Did not survive", 1: "Survived"}
    )
    results.to_csv(OUTPUT_PATH, index=False)
    print("BATCH INFERENCE — 20 records processed together")
    print(results.to_string(index=False))
    print(f"Saved {len(results)} predictions to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
