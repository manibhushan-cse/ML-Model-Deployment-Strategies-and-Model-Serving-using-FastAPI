"""Task 2: Predict one held-out passenger using the saved pipeline."""
from pathlib import Path
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
    # Recreate the deterministic hold-out partition used during training.
    _, X_test, _, _ = train_test_split(
        df[FEATURES], df["Survived"].astype(int),
        test_size=0.20, random_state=42, stratify=df["Survived"].astype(int)
    )
    passenger = X_test.iloc[[0]].copy()
    model = joblib.load(MODEL_PATH)
    prediction = int(model.predict(passenger)[0])
    passenger_id = int(df.loc[passenger.index[0], "PassengerId"]) if "PassengerId" in df else "unknown"
    label = "Survived" if prediction == 1 else "Did not survive"
    print("ONLINE INFERENCE")
    print(f"Passenger row index: {passenger.index[0]}")
    print(f"PassengerId: {passenger_id}")
    print(passenger.to_string(index=False))
    print(f"Prediction: {prediction} ({label})")

if __name__ == "__main__":
    main()
