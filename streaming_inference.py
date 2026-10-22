"""Task 4: Simulate streaming inference, one held-out record per second."""
from pathlib import Path
import time
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train.csv"
MODEL_PATH = ROOT / "model.pkl"
FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("model.pkl missing. Run: python train_model.py")
    df = pd.read_csv(DATA_PATH)
    _, X_test, _, _ = train_test_split(
        df[FEATURES], df["Survived"].astype(int),
        test_size=0.20, random_state=42, stratify=df["Survived"].astype(int)
    )
    if len(X_test) < 10:
        raise ValueError("At least 10 held-out records are required for streaming inference.")
    stream = X_test.iloc[:10]
    model = joblib.load(MODEL_PATH)
    print("STREAMING INFERENCE — one record every 1 second")
    for sequence, (idx, record) in enumerate(stream.iterrows(), start=1):
        prediction = int(model.predict(record.to_frame().T)[0])
        passenger_id = df.loc[idx, "PassengerId"] if "PassengerId" in df else idx
        label = "Survived" if prediction == 1 else "Did not survive"
        print(f"Incoming record {sequence}/10 | PassengerId={passenger_id} | Prediction={prediction} ({label})")
        if sequence < 10:
            time.sleep(1)

if __name__ == "__main__":
    main()
