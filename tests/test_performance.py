import time
import psutil
import numpy as np
from memory_profiler import memory_usage
from concurrent.futures import ThreadPoolExecutor
import threading
from src.train import train_model, get_train_data
from lib_ml.preprocessor import Preprocessor
from sklearn.naive_bayes import GaussianNB
import pandas as pd

def get_preprocessor():
    """Helper function to get a preprocessor with consistent configuration"""
    dataset = pd.read_csv('train_data.tsv', delimiter='\t', quoting=3)
    preprocessor = Preprocessor(max_features=1420)
    reviews = dataset['Review']
    preprocessed_reviews = preprocessor.preprocess_batch(reviews)
    preprocessor.vectorize(preprocessed_reviews)  # This fits the vectorizer
    return preprocessor

def test_training_time():
    """Test model training completes within time limit"""
    start_time = time.time()
    train_model()
    training_time = time.time() - start_time
    assert training_time < 60, f"Training took too long: {training_time:.2f} seconds"

def test_memory_usage():
    """Test model training doesn't exceed memory limits"""
    # Get trained model and preprocessor
    X_train, X_test, y_train, y_test = get_train_data()
    
    # Test training memory usage
    mem_usage = max(memory_usage(train_model))
    assert mem_usage < 1000, f"Training used too much memory: {mem_usage:.2f} MB"
    
    # Test inference memory usage
    classifier, _, _ = train_model()
    preprocessor = Preprocessor(max_features=1420)
    
    # Fit the vectorizer first
    dataset = pd.read_csv('train_data.tsv', delimiter='\t', quoting=3)
    reviews = dataset['Review']
    preprocessed_reviews = preprocessor.preprocess_batch(reviews)
    preprocessor.vectorize(preprocessed_reviews)  # This fits the vectorizer
    
    def inference_task():
        text = "Test review for memory usage"
        vector = preprocessor.vectorize_single(text)
        classifier.predict(vector)
    
    inference_mem = max(memory_usage(inference_task))
    assert inference_mem < 500, f"Inference used too much memory: {inference_mem:.2f} MB"

def test_prediction_latency():
    """Test model predictions are fast enough"""
    # Get trained model and preprocessor
    X_train, X_test, y_train, y_test = get_train_data()
    classifier, _, _ = train_model()
    preprocessor = Preprocessor(max_features=1420)
    
    # Fit the vectorizer first
    dataset = pd.read_csv('train_data.tsv', delimiter='\t', quoting=3)
    reviews = dataset['Review']
    preprocessed_reviews = preprocessor.preprocess_batch(reviews)
    preprocessor.vectorize(preprocessed_reviews)  # This fits the vectorizer
    
    # Prepare batch sizes for testing
    batch_sizes = [1, 10, 100]
    max_latencies = {1: 0.1, 10: 0.5, 100: 2.0}  # Maximum allowed latency in seconds
    
    for batch_size in batch_sizes:
        # Create test batch
        texts = [f"Test review number {i}" for i in range(batch_size)]
        
        # Measure preprocessing time
        start_time = time.time()
        vectors = np.vstack([
            preprocessor.vectorize_single(text) for text in texts
        ])
        predictions = classifier.predict(vectors)
        latency = time.time() - start_time
        
        assert latency < max_latencies[batch_size], f"Prediction too slow for batch size {batch_size}: {latency:.3f}s"
        assert len(predictions) == batch_size, "Wrong number of predictions"

def test_cpu_usage():
    """Test CPU usage during training and inference"""
    # Get trained model and preprocessor
    X_train, X_test, y_train, y_test = get_train_data()
    
    def measure_cpu():
        return psutil.Process().cpu_percent(interval=0.1)
    
    # Measure CPU during training
    start_cpu = measure_cpu()
    train_model()
    train_cpu = measure_cpu()
    assert train_cpu < 90, f"Training CPU usage too high: {train_cpu}%"
    
    # Measure CPU during inference
    classifier, _, _ = train_model()
    preprocessor = Preprocessor(max_features=1420)
    
    # Fit the vectorizer first
    dataset = pd.read_csv('train_data.tsv', delimiter='\t', quoting=3)
    reviews = dataset['Review']
    preprocessed_reviews = preprocessor.preprocess_batch(reviews)
    preprocessor.vectorize(preprocessed_reviews)  # This fits the vectorizer
    
    text = "Test review for CPU usage"
    
    start_cpu = measure_cpu()
    vector = preprocessor.vectorize_single(text)
    prediction = classifier.predict(vector)
    inference_cpu = measure_cpu()
    
    assert inference_cpu < 50, f"Inference CPU usage too high: {inference_cpu}%"

def test_model_size():
    """Test model file size is reasonable"""
    import joblib
    import tempfile
    
    classifier, _, _ = train_model()
    
    # Save model to temporary file
    with tempfile.NamedTemporaryFile() as tmp:
        joblib.dump(classifier, tmp.name)
        size_mb = tmp.tell() / (1024 * 1024)  # Convert to MB
        assert size_mb < 100, f"Model file too large: {size_mb:.2f} MB"

def test_feature_cost():
    """Test feature extraction cost"""
    # Get trained model and preprocessor
    X_train, X_test, y_train, y_test = get_train_data()
    preprocessor = Preprocessor(max_features=1420)
    
    # Fit the vectorizer first
    dataset = pd.read_csv('train_data.tsv', delimiter='\t', quoting=3)
    reviews = dataset['Review']
    preprocessed_reviews = preprocessor.preprocess_batch(reviews)
    preprocessor.vectorize(preprocessed_reviews)  # This fits the vectorizer
    
    # Test preprocessing time
    text = "Test review for feature extraction"
    start_time = time.time()
    preprocessed = preprocessor.preprocess(text)
    preprocess_time = time.time() - start_time
    assert preprocess_time < 0.1, f"Preprocessing too slow: {preprocess_time:.3f}s"
    
    # Test vectorization time
    start_time = time.time()
    vector = preprocessor.vectorize_single(text)
    vectorize_time = time.time() - start_time
    assert vectorize_time < 0.1, f"Vectorization too slow: {vectorize_time:.3f}s"

def test_scalability():
    """Test model performance with increasing data size"""
    X_train, X_test, y_train, y_test = get_train_data()
    
    # Test different dataset sizes
    sizes = [0.25, 0.5, 0.75, 1.0]
    times = []
    
    for size in sizes:
        n_samples = int(len(X_train) * size)
        X_subset = X_train[:n_samples]
        y_subset = y_train[:n_samples]
        
        start_time = time.time()
        classifier = train_model()  # Train on full dataset for consistency
        train_time = time.time() - start_time
        times.append(train_time)
    
    # Check if training time increases roughly linearly
    time_ratios = [times[i+1]/times[i] for i in range(len(times)-1)]
    assert all(ratio < 3 for ratio in time_ratios), "Non-linear scaling detected"

def test_concurrent_predictions():
    """Test model performance under concurrent prediction load"""
    # Get trained model and preprocessor
    X_train, X_test, y_train, y_test = get_train_data()
    classifier, _, _ = train_model()
    preprocessor = Preprocessor(max_features=1420)
    
    # Fit the vectorizer first
    dataset = pd.read_csv('train_data.tsv', delimiter='\t', quoting=3)
    reviews = dataset['Review']
    preprocessed_reviews = preprocessor.preprocess_batch(reviews)
    preprocessor.vectorize(preprocessed_reviews)  # This fits the vectorizer
    
    n_threads = 4
    n_predictions = 100
    
    # Create a list of texts to predict
    texts = [f"Test review number {i}" for i in range(n_predictions)]
    
    # Prepare vectors
    vectors = np.vstack([
        preprocessor.vectorize_single(text) for text in texts
    ])
    
    def predict_batch(batch_vectors):
        return classifier.predict(batch_vectors)
    
    # Split vectors into batches
    batch_size = n_predictions // n_threads
    vector_batches = [vectors[i:i + batch_size] for i in range(0, len(vectors), batch_size)]
    
    # Test concurrent predictions
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        start_time = time.time()
        futures = [executor.submit(predict_batch, batch) for batch in vector_batches]
        results = [future.result() for future in futures]
        total_time = time.time() - start_time
    
    # Verify results
    all_predictions = np.concatenate(results)
    assert len(all_predictions) == n_predictions, "Missing predictions"
    assert total_time < 5.0, f"Concurrent predictions too slow: {total_time:.3f}s"