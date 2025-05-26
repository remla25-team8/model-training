import joblib
import pytest
import os
import json
import tempfile
import pandas as pd
from huggingface_hub import HfApi
from pathlib import Path
from lib_ml.preprocessor import Preprocessor
from train import train
import numpy as np
from model_upload import upload_model
from evaluate import evaluate_model
import shutil

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


@pytest.fixture
def trained_model():
    """Fixture that provides a trained model, confusion matrix, and accuracy"""
    X_train = np.load('data/splits/X_train.npy')
    y_train = np.load('data/splits/y_train.npy')
    X_test = np.load('data/splits/X_test.npy')
    y_test = np.load('data/splits/y_test.npy')

    # Use only 500 samples for training since this is only for infrastructure.
    classifier = train(X_train[:500], y_train[:500])
    metrics = evaluate_model(classifier, X_test, y_test)
    cm = metrics['confusion_matrix']
    acc = metrics['accuracy']
    return classifier, cm, acc


def test_model_serialization(trained_model):
    """Test model can be serialized and deserialized"""
    # Get trained model and preprocessor
    classifier, _, _ = trained_model

    # Create temporary file path
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=True) as tmp:
        model_path = Path(tmp.name)

        # Test serialization
        joblib.dump(classifier, model_path)
        assert model_path.exists(), "Model file not created"
        assert model_path.stat().st_size > 0, "Model file is empty"

        # Test deserialization
        loaded_model = joblib.load(model_path)
        assert hasattr(loaded_model, 'predict'), "Loaded model missing predict method"

        # Test prediction consistency
        X_test = np.load('data/splits/X_test.npy')
        original_pred = classifier.predict(X_test)
        loaded_pred = loaded_model.predict(X_test)
        assert (original_pred == loaded_pred).all(), "Loaded model predictions differ"

        # File will be automatically deleted when exiting the with block


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
