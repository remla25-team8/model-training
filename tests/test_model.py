from src.train import train_model, get_data_splits
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from lib_ml.preprocessor import Preprocessor
import pandas as pd

def test_train_test_split():
    """Test that train-test split maintains class distribution"""
    X_train, X_test, y_train, y_test = get_data_splits()
    train_balance = np.mean(y_train)
    test_balance = np.mean(y_test)
    assert abs(train_balance - test_balance) < 0.1  # Similar balance
    
    # Test sizes
    assert len(X_train) > len(X_test), "Training set should be larger than test set"
    assert len(X_train) + len(X_test) > 0, "Data split should not be empty"

def test_model_training():
    """Test that model training produces valid outputs"""
    classifier, cm, acc = train_model()
    assert hasattr(classifier, 'predict')  # Is a trained sklearn model
    assert cm.shape == (2, 2)  # Binary classification confusion matrix
    assert 0 <= acc <= 1  # Valid accuracy score

def test_model_metrics():
    """Test model performance metrics meet minimum thresholds"""
    X_train, X_test, y_train, y_test = get_data_splits()
    classifier, _, acc = train_model()
    
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

def test_model_predictions():
    """Test model makes valid predictions"""
    # Get trained model and preprocessor
    classifier, _, _ = train_model()
    
    # Get a fitted preprocessor
    dataset = pd.read_csv('data/raw/train_data.tsv', delimiter='\t', quoting=3)
    preprocessor = Preprocessor(max_features=1420)
    reviews = dataset['Review']
    preprocessed_reviews = preprocessor.preprocess_batch(reviews)
    preprocessor.vectorize(preprocessed_reviews)  # This fits the vectorizer
    
    # Test cases
    test_texts = [
        "great amazing wonderful",
        "excellent service and food",
        "highly recommend this place",
        "terrible awful horrible",
        "worst experience ever",
        "would not recommend"
    ]
    
    # Get predictions
    vectors = np.vstack([
        preprocessor.vectorize_single(text) for text in test_texts
    ])
    predictions = classifier.predict(vectors)
    probabilities = classifier.predict_proba(vectors)
    
    # Test prediction validity
    assert predictions.shape == (len(test_texts),), "Wrong prediction shape"
    assert np.all(np.isin(predictions, [0, 1])), "Invalid prediction values"
    assert probabilities.shape == (len(test_texts), 2), "Wrong probability shape"
    assert np.allclose(np.sum(probabilities, axis=1), 1), "Probabilities don't sum to 1"
    assert np.all((0 <= probabilities) & (probabilities <= 1)), "Invalid probability values"

def test_model_calibration():
    """Test model probability calibration"""
    classifier, _, _ = train_model()
    X_train, X_test, y_train, y_test = get_data_splits()
    
    # Get probability predictions
    probas = classifier.predict_proba(X_test)
    
    # Test probability ranges
    assert np.all(probas >= 0) and np.all(probas <= 1), "Probabilities outside [0,1] range"
    assert np.allclose(np.sum(probas, axis=1), 1), "Probabilities don't sum to 1"
    
    # Test correlation between probabilities and actual outcomes
    pred_class_1_proba = probas[:, 1]
    correlation = np.corrcoef(pred_class_1_proba, y_test)[0, 1]
    assert correlation > 0.3, "Weak correlation between predictions and actual outcomes"