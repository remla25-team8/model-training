from train import train
from preprocess import get_data_splits
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from lib_ml.preprocessor import Preprocessor
import pandas as pd
import pytest
from evaluate import evaluate_model
import joblib
import json


@pytest.fixture
def get_splits():
    X_train = np.load('data/splits/X_train.npy')
    X_test = np.load('data/splits/X_test.npy')
    y_train = np.load('data/splits/y_train.npy')
    y_test = np.load('data/splits/y_test.npy')
    return X_train, X_test, y_train, y_test


@pytest.fixture
def trained_model():
    """Fixture that provides a trained model, confusion matrix, and accuracy"""
    # Load trained model
    classifier = joblib.load('models/sentiment_classifier.joblib')

    # Load metrics
    with open('metrics/metrics.json', 'r') as f:
        metrics = json.load(f)

    cm = metrics['confusion_matrix']
    acc = metrics['accuracy']

    return classifier, cm, acc


def test_train_test_split(get_splits):
    """Test that train-test split maintains class distribution"""
    X_train, X_test, y_train, y_test = get_splits
    train_balance = np.mean(y_train)
    test_balance = np.mean(y_test)
    assert abs(train_balance - test_balance) < 0.1  # Similar balance

    # Test sizes
    assert len(X_train) > len(X_test), "Training set should be larger than test set"
    assert len(X_train) + len(X_test) > 0, "Data split should not be empty"


def test_model_training(trained_model):
    """Test that model training produces valid outputs"""
    classifier, cm, acc = trained_model
    assert hasattr(classifier, 'predict')  # Is a trained sklearn model
    assert len(cm) == 2 and len(cm[0]) == 2  # Binary classification confusion matrix
    assert 0 <= acc <= 1  # Valid accuracy score


def test_model_metrics(get_splits, trained_model):
    """Test model performance metrics meet minimum thresholds"""
    X_train, X_test, y_train, y_test = get_splits
    classifier, _, _ = trained_model

    # Get predictions
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[:, 1]

    # Test various metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Define minimum thresholds with lower values
    assert auc > 0.65, "AUC-ROC score below threshold"
    assert precision > 0.6, "Precision score below threshold"
    assert recall > 0.5, "Recall score below threshold"


def test_model_calibration(get_splits, trained_model):
    """Test model probability calibration"""
    classifier, _, _ = trained_model
    X_train, X_test, y_train, y_test = get_splits

    # Get probability predictions
    probas = classifier.predict_proba(X_test)

    # Test probability ranges
    assert np.all(probas >= 0) and np.all(probas <= 1), "Probabilities outside [0,1] range"
    assert np.allclose(np.sum(probas, axis=1), 1), "Probabilities don't sum to 1"

    # Test correlation between probabilities and actual outcomes
    pred_class_1_proba = probas[:, 1]
    correlation = np.corrcoef(pred_class_1_proba, y_test)[0, 1]
    assert correlation > 0.3, "Weak correlation between predictions and actual outcomes"


def test_model_stability_across_samples(get_splits, trained_model):
    """Test model performance consistency across random test set samples"""
    X_train, X_test, y_train, y_test = get_splits
    classifier, _, baseline_acc = trained_model

    # Test on different random samples from test set
    for _ in range(3):  # Test multiple random samples
        # Randomly sample 25% of test data
        sample_indices = np.random.choice(len(X_test), size=int(0.25 * len(X_test)), replace=False)
        X_sample = X_test[sample_indices]
        y_sample = y_test[sample_indices]

        # Get predictions on the sample
        y_pred = classifier.predict(X_sample)
        sample_acc = np.mean(y_pred == y_sample)  # Calculate accuracy directly

        # Allow for some variance but catch significant instability
        assert abs(sample_acc - baseline_acc) < 0.2, f"Model performance unstable: baseline={baseline_acc:.3f}, sample={sample_acc:.3f}"
