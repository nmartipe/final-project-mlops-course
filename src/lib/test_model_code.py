import os
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from your_module import save_model_and_encoder, train_model, compute_model_metrics

@pytest.fixture
def sample_data():
    np.random.seed(42)
    X = pd.DataFrame(np.random.rand(100, 5), columns=['feat1', 'feat2', 'feat3', 'feat4', 'feat5'])
    y = np.random.randint(2, size=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def test_save_model_and_encoder(sample_data):
    X_train, _, _, _ = sample_data
    model = RandomForestClassifier(random_state=42)
    encoder = OneHotEncoder()
    model_path = 'test_model.pkl'
    encoder_path = 'test_encoder.pkl'

    save_model_and_encoder(model, model_path, encoder, encoder_path)

    assert os.path.exists(model_path)
    assert os.path.exists(encoder_path)

    os.remove(model_path)
    os.remove(encoder_path)

def test_train_model(sample_data):
    X_train, _, y_train, _ = sample_data
    trained_model = train_model(X_train, y_train)
    assert trained_model is not None

def test_compute_model_metrics(sample_data):
    _, X_test, _, y_test = sample_data
    trained_model = train_model(X_test, y_test)
    preds = trained_model.predict(X_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert fbeta >= 0.0
