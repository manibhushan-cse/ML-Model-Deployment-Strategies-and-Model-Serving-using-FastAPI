"""Task 1: Train, evaluate, and save one reusable Titanic ML pipeline."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train.csv"
MODEL_PATH = ROOT / "model.pkl"
METRICS_PATH = ROOT / "reports" / "evaluation.txt"

TARGET = "Survived"
NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Download train.csv from https://www.kaggle.com/competitions/titanic/data "
            "and place it inside the project's data/ folder."
        )

    df = pd.read_csv(DATA_PATH)
    required = set(FEATURES + [TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"train.csv is missing required columns: {sorted(missing)}")
    df = df.dropna(subset=[TARGET]).copy()
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    # Keep a held-out test split for a fair accuracy estimate.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # Save the complete preprocessing + model pipeline, not only the classifier.
    joblib.dump(pipeline, MODEL_PATH)

    report = (
        "TITANIC MODEL EVALUATION\n"
        f"Dataset rows used: {len(df)}\n"
        f"Training rows: {len(X_train)}\n"
        f"Testing rows: {len(X_test)}\n"
        f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n"
        "Confusion matrix (rows=true labels, columns=predicted labels):\n"
        f"{confusion_matrix(y_test, predictions)}\n\n"
        "Classification report:\n"
        f"{classification_report(y_test, predictions, zero_division=0)}\\n"
    )
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved reusable pipeline to: {MODEL_PATH}")
    print(f"Saved evaluation details to: {METRICS_PATH}")

if __name__ == "__main__":
    main()
