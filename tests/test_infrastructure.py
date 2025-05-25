import joblib
import pytest
import os
import json
import tempfile
import pandas as pd
from src.train import upload_model, train_model, get_data_splits
from huggingface_hub import HfApi
from pathlib import Path
from lib_ml.preprocessor import Preprocessor

def test_model_serialization(tmp_path):
    """Test model can be serialized and deserialized"""
    # Get trained model and preprocessor
    X_train, X_test, y_train, y_test = get_data_splits()
    classifier, _, _ = train_model()
    model_path = tmp_path / "model.joblib"
    
    # Test serialization
    joblib.dump(classifier, model_path)
    assert model_path.exists(), "Model file not created"
    assert model_path.stat().st_size > 0, "Model file is empty"
    
    # Test deserialization
    loaded_model = joblib.load(model_path)
    assert hasattr(loaded_model, 'predict'), "Loaded model missing predict method"
    
    # Test prediction consistency
    dataset = pd.read_csv('data/raw/train_data.tsv', delimiter='\t', quoting=3)
    preprocessor = Preprocessor(max_features=1420)
    reviews = dataset['Review']
    preprocessed_reviews = preprocessor.preprocess_batch(reviews)
    preprocessor.vectorize(preprocessed_reviews)  # This fits the vectorizer
    
    test_text = "This is a test review"
    vector = preprocessor.vectorize_single(test_text)
    
    original_pred = classifier.predict(vector)
    loaded_pred = loaded_model.predict(vector)
    assert (original_pred == loaded_pred).all(), "Loaded model predictions differ"

def test_model_metadata():
    """Test model metadata generation and format"""
    classifier, cm, acc = train_model()
    metadata = {
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "model_type": "GaussianNB",
        "task": "sentiment_analysis",
        "version": "test"
    }
    
    # Test metadata format
    assert isinstance(metadata["accuracy"], float)
    assert isinstance(metadata["confusion_matrix"], list)
    assert isinstance(metadata["model_type"], str)
    assert isinstance(metadata["task"], str)
    assert isinstance(metadata["version"], str)
    
    # Test metadata values
    assert 0 <= metadata["accuracy"] <= 1
    assert len(metadata["confusion_matrix"]) == 2  # Binary classification
    assert metadata["model_type"] == "GaussianNB"
    assert metadata["task"] == "sentiment_analysis"

def test_model_artifacts():
    """Test model artifacts structure and content"""
    # Get trained model and preprocessor
    X_train, X_test, y_train, y_test = get_data_splits()
    
    # Create temporary directory for artifacts
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        os.makedirs(temp_path / "model", exist_ok=True)
        
        # Train and save model
        classifier, cm, acc = train_model()
        
        # Save model and metadata
        model_path = temp_path / "model" / "sentiment_classifier.joblib"
        metadata_path = temp_path / "model" / "metadata.json"
        
        joblib.dump(classifier, model_path)
        metadata = {
            "accuracy": float(acc),
            "confusion_matrix": cm.tolist(),
            "model_type": "GaussianNB",
            "task": "sentiment_analysis",
            "version": "test"
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        # Test artifact structure
        assert model_path.exists(), "Model file not created"
        assert metadata_path.exists(), "Metadata file not created"
        
        # Test model loading
        loaded_model = joblib.load(model_path)
        assert hasattr(loaded_model, 'predict'), "Invalid model file"
        
        # Test metadata loading
        with open(metadata_path, 'r') as f:
            loaded_metadata = json.load(f)
        assert loaded_metadata == metadata, "Metadata mismatch"
        
        # Test prediction consistency
        dataset = pd.read_csv('data/raw/train_data.tsv', delimiter='\t', quoting=3)
        preprocessor = Preprocessor(max_features=1420)
        reviews = dataset['Review']
        preprocessed_reviews = preprocessor.preprocess_batch(reviews)
        preprocessor.vectorize(preprocessed_reviews)  # This fits the vectorizer
        
        test_text = "This is a test review"
        vector = preprocessor.vectorize_single(test_text)
        
        original_pred = classifier.predict(vector)
        loaded_pred = loaded_model.predict(vector)
        assert (original_pred == loaded_pred).all(), "Loaded model predictions differ"

def test_environment_variables():
    """Test environment variable handling"""
    # Test missing token
    if "HF_TOKEN" in os.environ:
        token = os.environ["HF_TOKEN"]
        del os.environ["HF_TOKEN"]
        try:
            with pytest.raises(ValueError):
                classifier, cm, acc = train_model()
                upload_model(classifier, cm, acc, version="test")
        finally:
            os.environ["HF_TOKEN"] = token
    else:
        with pytest.raises(ValueError):
            classifier, cm, acc = train_model()
            upload_model(classifier, cm, acc, version="test")

@pytest.mark.skipif(not os.getenv("HF_TOKEN"), reason="Hugging Face token not available")
def test_model_upload():
    """Test model can be uploaded to registry (integration test)"""
    classifier, cm, acc = train_model()
    version = "test"
    
    # Test upload
    try:
        upload_model(classifier, cm, acc, version=version)
    except Exception as e:
        pytest.fail(f"Model upload failed: {str(e)}")
    
    # Test model exists in registry
    api = HfApi()
    repo_name = "todor-cmd/sentiment-classifier"
    assert api.repo_exists(repo_name), "Model repository not found"
    assert api.revision_exists(repo_name, version), "Model version not found"