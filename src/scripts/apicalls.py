import requests

API_URL = 'https://final-project-mlops-course.onrender.com'

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

response = requests.post(API_URL + "/predict", json=data)
print("The prediction is:")
print(response.json())