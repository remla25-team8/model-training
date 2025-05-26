import json
import numpy as np
from datetime import datetime
from old_train import train_model, get_data_splits
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
import logging
import tempfile
from pathlib import Path
from lib_ml.preprocessor import Preprocessor
import time
import os

def test_metadata_generation():
    """Test metadata generation and logging"""
    classifier, cm, acc = train_model()
    
    # Test confusion matrix properties
    assert isinstance(cm, np.ndarray), "Confusion matrix should be numpy array"
    assert cm.shape == (2, 2), "Confusion matrix should be 2x2 for binary classification"
    assert np.all(cm >= 0), "Confusion matrix should have non-negative values"
    
    # Test accuracy properties
    assert isinstance(acc, (float, np.float64)), "Accuracy should be float"
    assert 0 <= acc <= 1, "Accuracy should be between 0 and 1"

def test_performance_regression():
    """Test for performance regression"""
    # Get baseline performance
    X_train, X_test, y_train, y_test = get_data_splits()
    classifier, _, acc = train_model()
    
    # Test on different data splits
    for _ in range(3):  # Test multiple splits
        X_train, X_test, y_train, y_test = get_data_splits()
        y_pred = classifier.predict(X_test)
        split_acc = accuracy_score(y_test, y_pred)
        
        # Allow for some variance but catch significant regressions
        assert abs(split_acc - acc) < 0.1, "Large performance variance between splits"

def test_prediction_monitoring():
    """Test prediction monitoring capabilities"""
    # Get trained model and preprocessor
    X_train, X_test, y_train, y_test = get_data_splits()
    classifier, _, _ = train_model()
    
    # Get a fitted preprocessor
    dataset = pd.read_csv('data/raw/train_data.tsv', delimiter='\t', quoting=3)
    preprocessor = Preprocessor(max_features=1420)
    reviews = dataset['Review']
    preprocessed_reviews = preprocessor.preprocess_batch(reviews)
    preprocessor.vectorize(preprocessed_reviews)  # This fits the vectorizer
    
    # Setup logging with a stream handler
    log_file = tempfile.NamedTemporaryFile(delete=False, suffix='.log')
    logger = logging.getLogger('prediction_monitor')
    logger.setLevel(logging.INFO)
    
    # Add both file and stream handlers
    file_handler = logging.FileHandler(log_file.name)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    try:
        # Test sample predictions with monitoring
        sample_texts = [
            "This restaurant was amazing!",
            "Terrible service and food",
            "Average experience, nothing special"
        ]
        
        for text in sample_texts:
            # Log prediction request
            logger.info(f"Prediction request - Input: {text}")
            
            # Make prediction
            vector = preprocessor.vectorize_single(text)
            prediction = classifier.predict(vector)
            probabilities = classifier.predict_proba(vector)
            
            # Log prediction results
            logger.info(f"Prediction: {prediction[0]}, Confidence: {max(probabilities[0]):.2f}")
            
            # Basic validity checks
            assert prediction.shape == (1,), "Invalid prediction shape"
            assert probabilities.shape == (1, 2), "Invalid probability shape"
            assert np.allclose(np.sum(probabilities, axis=1), 1), "Probabilities don't sum to 1"
            
        # Check log file exists and contains entries
        assert os.path.exists(log_file.name), "Log file not created"
        with open(log_file.name, 'r') as f:
            log_content = f.read()
            assert "Prediction request" in log_content, "Missing prediction request logs"
            assert "Prediction:" in log_content, "Missing prediction result logs"
            
    finally:
        # Cleanup
        logger.removeHandler(file_handler)
        file_handler.close()
        os.unlink(log_file.name)

def test_data_drift_detection():
    """Test data drift detection capabilities"""
    X_train, X_test, y_train, y_test = get_data_splits()
    classifier, _, _ = train_model()
    
    # Function to compute basic distribution statistics
    def get_distribution_stats(X):
        return {
            'mean': np.mean(X),
            'std': np.std(X),
            'min': np.min(X),
            'max': np.max(X)
        }
    
    # Get training data statistics
    train_stats = get_distribution_stats(X_train)
    
    # Compare with test data statistics
    test_stats = get_distribution_stats(X_test)
    
    # Check for significant distribution shifts
    for metric in ['mean', 'std']:
        relative_diff = abs(train_stats[metric] - test_stats[metric]) / train_stats[metric]
        assert relative_diff < 0.5, f"Significant {metric} shift detected"

def test_model_versioning():
    """Test model versioning and tracking"""
    # Train multiple model versions
    models = []
    accuracies = []
    
    for _ in range(3):  # Train 3 versions
        classifier, _, acc = train_model()
        models.append(classifier)
        accuracies.append(acc)
    
    # Test version tracking
    assert len(models) == 3, "Failed to track all model versions"
    assert len(accuracies) == 3, "Failed to track all accuracy scores"
    
    # Test performance consistency
    acc_std = np.std(accuracies)
    assert acc_std < 0.1, "High variance in model performance across versions"
    
    # Test prediction consistency
    X_train, X_test, y_train, y_test = get_data_splits()
    base_predictions = models[0].predict(X_test)
    
    for model in models[1:]:
        current_predictions = model.predict(X_test)
        agreement = np.mean(base_predictions == current_predictions)
        assert agreement > 0.8, "Low prediction consistency between model versions"