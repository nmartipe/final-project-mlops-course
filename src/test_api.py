# test_app.py
import requests

def test_read_root():
    response = requests.get("http://127.0.0.1:8000/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the FastAPI Model Inference API!"}

def test_predict_below_50k():
    # Test for the ML model's prediction when the income is below 50k
    data = {
        "age": 15,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    response = requests.post("http://127.0.0.1:8000/predict", json=data)
    assert response.status_code == 200
    result = response.json()
    assert "predictions" in result
    assert len(result["predictions"]) == 1
    assert result["predictions"][0] == 0

def test_predict_above_50k():
    # Test for the ML model's prediction when the income is above 50k
    data = {
        "age": 45,
        "workclass": "Private",
        "fnlgt": 154374,
        "education": "HS-grad",
        "education-num": 9,
        "marital-status": "Married-civ-spouse",
        "occupation": "Machine-op-inspct",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 50000,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States",
    }
    
    response = requests.post("http://127.0.0.1:8000/predict", json=data)
    assert response.status_code == 200
    result = response.json()
    assert "predictions" in result
    assert len(result["predictions"]) == 1
    assert result["predictions"][0] == 1
