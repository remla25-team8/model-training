import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
from lib_ml.preprocessor import Preprocessor
import pandas as pd
import pytest
import joblib
import json
import logging

@pytest.fixture
def dataset():
    """Fixture that provides the training dataset"""
    return pd.read_csv('data/raw/raw_data.tsv', delimiter='\t', quoting=3)


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

@pytest.fixture
def preprocessor(dataset):
    """Fixture that provides a fitted preprocessor"""
    preprocessor = Preprocessor(max_features=1420)
    reviews = dataset['Review']
    preprocessed_reviews = preprocessor.preprocess_batch(reviews)
    preprocessor.vectorize(preprocessed_reviews)  
    return preprocessor

def test_quality_on_slices(dataset, trained_model, preprocessor):
    """Test model performance on specific data slices"""
    classifier, _, _ = trained_model  # <-- Add this line to unpack the tuple
    
    def evaluate_slice(slice_data, slice_name):
        if len(slice_data) > 0:
            # Use vectorize_single for each review
            vectors = np.vstack([
                preprocessor.vectorize_single(text) for text in slice_data['Review']
            ])
            y = slice_data['Liked']
            acc = accuracy_score(y, classifier.predict(vectors))  # <-- Use classifier instead of trained_model
            assert acc > 0.6, f"Poor performance on {slice_name} slice (acc={acc:.2f})"
            return acc
        return None

    # Test on review length slices
    short_reviews = dataset[dataset['Review'].str.len() < 50]
    medium_reviews = dataset[(dataset['Review'].str.len() >= 50) & (dataset['Review'].str.len() < 200)]
    long_reviews = dataset[dataset['Review'].str.len() >= 200]

    results = {
        'short_reviews': evaluate_slice(short_reviews, "short reviews"),
        'medium_reviews': evaluate_slice(medium_reviews, "medium reviews"),
        'long_reviews': evaluate_slice(long_reviews, "long reviews")
    }

    # Log results
    for slice_name, acc in results.items():
        if acc is not None:
            logging.info(f"Accuracy on {slice_name}: {acc:.2f}")

def test_baseline_model(get_splits, trained_model):
    """Test model performance metrics meet minimum thresholds"""
    _, X_test, _, y_test = get_splits
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
    _, X_test, _, y_test = get_splits

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
    _, X_test, _, y_test = get_splits
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
