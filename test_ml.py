import pytest
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, inference, compute_model_metrics


@pytest.fixture
def sample_data():
    """Create small sample training data for tests."""
    np.random.seed(42)
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    return X, y


def test_train_model(sample_data):
    """Test that train_model returns a RandomForestClassifier."""
    X, y = sample_data
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)


def test_inference(sample_data):
    """Test that inference returns an array of the same length as input."""
    X, y = sample_data
    model = train_model(X, y)
    preds = inference(model, X)
    assert len(preds) == len(X)


def test_compute_model_metrics(sample_data):
    """Test that compute_model_metrics returns three floats between 0 and 1."""
    X, y = sample_data
    model = train_model(X, y)
    preds = inference(model, X)
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= fbeta <= 1.0