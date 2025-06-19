import joblib
import pytest
import os
import json
import tempfile
from huggingface_hub import HfApi
from pathlib import Path
from model_upload import upload_model
import shutil
import numpy as np
from sklearn.base import BaseEstimator

from dotenv import load_dotenv
from train import train
from preprocess import get_data_splits, preprocess_data
from evaluate import evaluate_model
import pandas as pd
load_dotenv('.secrets')

# Global variable to store zip path
zip_path = None


@pytest.fixture(scope="session", autouse=True)
def setup_teardown_zip():
    """Create and cleanup zip file for all tests"""
    global zip_path

    # Create a temporary directory for our zip file
    temp_dir = tempfile.mkdtemp()
    zip_path = Path(temp_dir) / "model.zip"

    # Create dummy model files
    model_dir = Path(temp_dir) / "model"
    model_dir.mkdir()

    # Create dummy model file
    dummy_model = {"dummy": "model"}
    joblib.dump(dummy_model, model_dir / "model.joblib")

    # Create dummy metrics file
    dummy_metrics = {"accuracy": 0.95, "confusion_matrix": [[10, 2], [3, 15]]}
    with open(model_dir / "metrics.json", "w") as f:
        json.dump(dummy_metrics, f)

    # Create zip file
    shutil.make_archive(str(zip_path.with_suffix("")), 'zip', model_dir)

    yield  # This is where the tests run

    # Cleanup after all tests
    if zip_path.exists():
        zip_path.unlink()
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="session")
def test_data_dir():
    """Create a temporary directory for test data"""    
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    
    # Cleanup after all tests
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
    
    return raw_data_path, df

@pytest.fixture(scope="session", autouse=True)
def cleanup_hf_revision():
    """Cleanup Hugging Face test revision before and after tests"""
    if os.getenv("HF_TOKEN"):
        api = HfApi()
        repo_name = "todor-cmd/sentiment-classifier"
        test_revision = "test"

        # Cleanup before tests
        try:
            if api.revision_exists(repo_name, test_revision):
                api.delete_revision(repo_name, test_revision)
        except Exception as e:
            print(f"Warning: Could not cleanup test revision before tests: {e}")

        yield  # This is where the tests run

        # Cleanup after tests
        try:
            if api.revision_exists(repo_name, test_revision):
                api.delete_revision(repo_name, test_revision)
        except Exception as e:
            print(f"Warning: Could not cleanup test revision after tests: {e}")

def test_unit_test_model(small_raw_dataset):
    """Test unit test model components with proper assertions"""
    raw_data_path, df = small_raw_dataset
    
    # Test get_data_splits
    X_train, X_test, y_train, y_test = get_data_splits(raw_data_path)
    
    # Assertions for data splits
    assert isinstance(X_train, np.ndarray), "X_train should be a numpy array"
    assert isinstance(X_test, np.ndarray), "X_test should be a numpy array"
    assert isinstance(y_train, pd.Series), "y_train should be a pandas Series"
    assert isinstance(y_test, pd.Series), "y_test should be a pandas Series"
    
    # Check shapes are consistent
    assert X_train.shape[0] == y_train.shape[0], "X_train and y_train should have same number of samples"
    assert X_test.shape[0] == y_test.shape[0], "X_test and y_test should have same number of samples"
    assert X_train.shape[1] == X_test.shape[1], "X_train and X_test should have same number of features"
    
    # Check that we have both train and test data
    assert X_train.shape[0] > 0, "Training set should not be empty"
    assert X_test.shape[0] > 0, "Test set should not be empty"
    
    # Check that labels are binary (0 or 1)
    assert set(y_train.unique()).issubset({0, 1}), "y_train should contain only 0s and 1s"
    assert set(y_test.unique()).issubset({0, 1}), "y_test should contain only 0s and 1s"
    
    # Test preprocess_data
    X, y = preprocess_data(df)
    
    # Assertions for preprocessing
    assert isinstance(X, np.ndarray), "Preprocessed X should be a numpy array"
    assert isinstance(y, pd.Series), "Preprocessed y should be a pandas Series"
    assert X.shape[0] == len(df), "Preprocessed data should have same number of samples as input"
    assert X.shape[0] == y.shape[0], "X and y should have same number of samples"
    assert X.shape[1] > 0, "Features should have at least one dimension"
    
    # Test train function
    model = train(X_train, y_train)
    
    # Assertions for model training
    assert model is not None, "Model should not be None"
    assert isinstance(model, BaseEstimator), "Model should be a sklearn estimator"
    assert hasattr(model, 'predict'), "Model should have predict method"
    assert hasattr(model, 'fit'), "Model should have fit method"
    
    # Test that model can make predictions
    predictions = model.predict(X_test)
    assert isinstance(predictions, np.ndarray), "Predictions should be numpy array"
    assert predictions.shape[0] == X_test.shape[0], "Should have one prediction per test sample"
    assert set(np.unique(predictions)).issubset({0, 1}), "Predictions should be binary (0 or 1)"
    
    # Test evaluate_model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Assertions for evaluation
    assert isinstance(metrics, dict), "Metrics should be a dictionary"
    assert 'accuracy' in metrics, "Metrics should contain accuracy"
    assert 'confusion_matrix' in metrics, "Metrics should contain confusion_matrix"
    
    # Check accuracy is valid
    assert isinstance(metrics['accuracy'], float), "Accuracy should be a float"
    assert 0.0 <= metrics['accuracy'] <= 1.0, "Accuracy should be between 0 and 1"
    
    # Check confusion matrix is valid
    cm = metrics['confusion_matrix']
    assert isinstance(cm, list), "Confusion matrix should be a list"
    assert len(cm) == 2, "Confusion matrix should be 2x2 for binary classification"
    assert all(len(row) == 2 for row in cm), "Each row in confusion matrix should have 2 elements"
    assert all(isinstance(val, int) for row in cm for val in row), "Confusion matrix values should be integers"
    assert sum(sum(row) for row in cm) == len(y_test), "Confusion matrix should sum to number of test samples"

def test_environment_variables():
    """Test missing environment variable handling"""
    if "HF_TOKEN" in os.environ:
        token = os.environ["HF_TOKEN"]
        del os.environ["HF_TOKEN"]
        try:
            with pytest.raises(ValueError):
                upload_model(zip_path, version="test")
        finally:
            os.environ["HF_TOKEN"] = token
    else:
        with pytest.raises(ValueError):
            upload_model(zip_path, version="test")


@pytest.mark.skipif(not os.getenv("HF_TOKEN"), reason="Hugging Face token not available")
def test_model_upload():
    """Test model can be uploaded to registry (integration test)"""

    # Test upload
    try:
        upload_model(zip_path, version="test")
    except Exception as e:
        pytest.fail(f"Model upload failed: {str(e)}")

    # Test model exists in registry
    api = HfApi()
    repo_name = "todor-cmd/sentiment-classifier"
    assert api.repo_exists(repo_name), "Model repository not found"
    assert api.revision_exists(repo_name, "test"), "Model version not found"



