"""Python client to test the FastAPI /predict endpoint."""
import requests

API_URL = "http://127.0.0.1:8000"

def main():
    payload = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S"
    }
    response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
    print("HTTP status:", response.status_code)
    print("Response:", response.text)
    response.raise_for_status()

if __name__ == "__main__":
    main()
