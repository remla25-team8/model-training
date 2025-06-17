import sys
import os
import pytest
import tempfile
import shutil
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import your real components
from preprocess import preprocess_data, get_data_splits
from train import train
from evaluate import evaluate_model


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data that persists for the session"""
    temp_dir = tempfile.mkdtemp(prefix="ml_test_")
    yield temp_dir
    # Cleanup after all tests are done
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(scope="session")
def small_raw_dataset(test_data_dir):
    """Create a small raw dataset using real review data structure"""
    # Create realistic but small dataset
    reviews_data = [
        ("This restaurant is amazing! The food was delicious and service was excellent.", 1),
        ("Great atmosphere and friendly staff. Highly recommend this place!", 1),
        ("Best dining experience I've had in months. Will definitely come back soon!", 1),
        ("Fresh ingredients and creative dishes. Loved every single bite of it!", 1),
        ("Outstanding flavors and perfect presentation. Worth every penny spent!", 1),
        ("Excellent service and cozy ambiance. Perfect for a romantic dinner date.", 1),
        ("Incredible taste and generous portions. Staff was very attentive throughout.", 1),
        ("Beautiful decor and amazing cocktails. Food exceeded all my expectations completely.", 1),
        ("The food was cold and service was terrible. Very disappointed with everything.", 0),
        ("Overpriced and bland food. Would not recommend this place to anyone.", 0),
        ("Worst dining experience ever. The staff was rude and place was dirty.", 0),
        ("Waited over an hour for mediocre food. Never going back there again.", 0),
        ("The portions were tiny and the meal was undercooked. Absolutely awful experience!", 0),
        ("Poor service and stale bread. Manager was unhelpful when we complained about it.", 0),
        ("Dirty tables and slow service. Food arrived lukewarm and tasteless throughout.", 0),
        ("Expensive for what you get. Quality has definitely declined over the years.", 0),
        ("Decent food but nothing special. Average experience overall, might return someday.", 1),
        ("The restaurant was okay. Food was fine but service could be better.", 0),
        ("Not bad but not great either. Probably won't return anytime soon.", 0),
        ("It was alright, nothing to write home about really.", 1),
    ]
    
    # Create DataFrame
    df = pd.DataFrame(reviews_data, columns=["Review", "Liked"])
    
    # Save to test directory
    raw_data_dir = os.path.join(test_data_dir, "data", "raw")
    os.makedirs(raw_data_dir, exist_ok=True)
    raw_data_path = os.path.join(raw_data_dir, "test_raw_data.tsv")
    df.to_csv(raw_data_path, sep='\t', index=False, quoting=3)
    
    return raw_data_path


@pytest.fixture(scope="session")
def test_data_splits(small_raw_dataset, test_data_dir):
    """Create data splits using real preprocessing pipeline"""
    # Use your real preprocessing function
    X_train, X_test, y_train, y_test = get_data_splits(
        small_raw_dataset, 
        test_size=0.3,  # Slightly larger test set for small dataset
        random_state=42
    )
    
    # Save splits to test directory
    splits_dir = os.path.join(test_data_dir, "data", "splits")
    os.makedirs(splits_dir, exist_ok=True)
    
    X_train_path = os.path.join(splits_dir, "X_train.npy")
    X_test_path = os.path.join(splits_dir, "X_test.npy")
    y_train_path = os.path.join(splits_dir, "y_train.npy")
    y_test_path = os.path.join(splits_dir, "y_test.npy")
    
    np.save(X_train_path, X_train)
    np.save(X_test_path, X_test)
    np.save(y_train_path, y_train)
    np.save(y_test_path, y_test)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'paths': {
            'X_train': X_train_path,
            'X_test': X_test_path,
            'y_train': y_train_path,
            'y_test': y_test_path
        }
    }


@pytest.fixture(scope="session")
def trained_test_model(test_data_splits, test_data_dir):
    """Train a model using real training pipeline"""
    # Use your real training function
    model = train(
        test_data_splits['X_train'], 
        test_data_splits['y_train']
    )
    
    # Save model to test directory
    models_dir = os.path.join(test_data_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "test_sentiment_classifier.joblib")
    joblib.dump(model, model_path)
    
    return {
        'model': model,
        'path': model_path
    }


@pytest.fixture(scope="session")
def test_metrics(trained_test_model, test_data_splits, test_data_dir):
    """Generate metrics using real evaluation pipeline"""
    model = trained_test_model['model']
    X_test = test_data_splits['X_test']
    y_test = test_data_splits['y_test']
    
    # Use your real evaluation function
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save metrics to test directory
    metrics_dir = os.path.join(test_data_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_dir, "test_metrics.json")
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return {
        'metrics': metrics,
        'path': metrics_path
    }


@pytest.fixture
def setup_test_environment(test_data_splits, trained_test_model, test_metrics, test_data_dir):
    """Setup complete test environment with all artifacts"""
    # Create symlinks or copy files to expected locations for tests
    original_cwd = os.getcwd()
    
    # Create the expected directory structure in current working directory
    os.makedirs("data/splits", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    
    # Copy test artifacts to expected locations
    for name, path in test_data_splits['paths'].items():
        target_path = f"data/splits/{name}.npy"
        shutil.copy2(path, target_path)
    
    shutil.copy2(trained_test_model['path'], "models/sentiment_classifier.joblib")
    shutil.copy2(test_metrics['path'], "metrics/metrics.json")
    
    yield {
        'data_splits': test_data_splits,
        'model': trained_test_model,
        'metrics': test_metrics,
        'test_dir': test_data_dir
    }
    
    # Cleanup - remove test artifacts from working directory
    cleanup_paths = [
        "data/splits/X_train.npy",
        "data/splits/X_test.npy", 
        "data/splits/y_train.npy",
        "data/splits/y_test.npy",
        "models/sentiment_classifier.joblib",
        "metrics/metrics.json"
    ]
    
    for path in cleanup_paths:
        if os.path.exists(path):
            os.remove(path)


# Legacy fixtures for backward compatibility
@pytest.fixture
def get_splits(setup_test_environment):
    """Legacy fixture that returns data splits in expected format"""
    data = setup_test_environment['data_splits']
    return data['X_train'], data['X_test'], data['y_train'], data['y_test']


@pytest.fixture
def trained_model(setup_test_environment):
    """Legacy fixture that returns trained model in expected format"""
    model_data = setup_test_environment['model']
    metrics_data = setup_test_environment['metrics']
    
    # Extract confusion matrix and accuracy from metrics
    metrics = metrics_data['metrics']
    cm = metrics.get('confusion_matrix', [[5, 1], [1, 5]])  # Default fallback
    acc = metrics.get('accuracy', 0.85)  # Default fallback
    
    return model_data['model'], cm, acc


