import numpy as np
from train import train
import pandas as pd
import logging
import tempfile
from lib_ml.preprocessor import Preprocessor
import os
import pytest
from evaluate import evaluate_model
from sklearn.metrics import accuracy_score
from preprocess import preprocess_data, get_data_splits


@pytest.fixture
def get_splits():
    X_train = np.load('data/splits/X_train.npy')
    X_test = np.load('data/splits/X_test.npy')
    y_train = np.load('data/splits/y_train.npy')
    y_test = np.load('data/splits/y_test.npy')
    return X_train, X_test, y_train, y_test


@pytest.fixture
def raw_dataset():
    """Load raw dataset for invariant testing"""
    return pd.read_csv('data/raw/raw_data.tsv', delimiter='\t', quoting=3)


def test_monitor_data_invariants(raw_dataset, get_splits):
    """Monitor data invariants: schema, label distribution, and train-test consistency"""
    X_train, X_test, y_train, y_test = get_splits
    
    # 1. SCHEMA INVARIANTS
    # Check required columns exist
    assert 'Review' in raw_dataset.columns, "Missing required column: Review"
    assert 'Liked' in raw_dataset.columns, "Missing required column: Liked"
    
    # Check data types
    assert raw_dataset['Review'].dtype == object, f"Review column should be string, got {raw_dataset['Review'].dtype}"
    assert pd.api.types.is_integer_dtype(raw_dataset['Liked']), f"Liked column should be integer, got {raw_dataset['Liked'].dtype}"
    
    # Check for null values
    assert raw_dataset['Review'].notna().all(), "Found null reviews"
    assert raw_dataset['Liked'].notna().all(), "Found null labels"
    
    # Check dataset not empty
    assert len(raw_dataset) > 0, "Dataset is empty"
    
    # Check no empty reviews
    assert not (raw_dataset['Review'].str.strip() == '').any(), "Found empty reviews"
    
    
    # 2. LABEL DISTRIBUTION INVARIANTS
    # Check training labels
    unique_train_labels = np.unique(y_train)
    assert len(unique_train_labels) >= 2, f"Training set should have at least 2 classes, got {len(unique_train_labels)}"
    assert set(unique_train_labels) == {0, 1}, f"Training labels should be {{0, 1}}, got {set(unique_train_labels)}"
    
    # Check test labels
    unique_test_labels = np.unique(y_test)
    assert len(unique_test_labels) >= 2, f"Test set should have at least 2 classes, got {len(unique_test_labels)}"
    assert set(unique_test_labels) == {0, 1}, f"Test labels should be {{0, 1}}, got {set(unique_test_labels)}"
    
    # Check minimum samples per class in training
    for label in [0, 1]:
        train_count = np.sum(y_train == label)
        assert train_count >= 5, f"Training class {label} has only {train_count} samples (minimum: 5)"
    
    # Check minimum samples per class in test
    for label in [0, 1]:
        test_count = np.sum(y_test == label)
        assert test_count >= 2, f"Test class {label} has only {test_count} samples (minimum: 2)"
    
    # Check for severe class imbalance
    train_balance = np.mean(y_train)
    test_balance = np.mean(y_test)
    assert 0.05 <= train_balance <= 0.95, f"Training set severely imbalanced: {train_balance:.3f} positive class"
    assert 0.05 <= test_balance <= 0.95, f"Test set severely imbalanced: {test_balance:.3f} positive class"
    
    # Check raw dataset labels are valid
    assert raw_dataset['Liked'].isin([0, 1]).all(), "Raw dataset contains invalid label values"
    
    
    # 3. TRAIN-TEST CONSISTENCY INVARIANTS
    # Feature dimension consistency
    assert X_train.shape[1] == X_test.shape[1], f"Feature dimension mismatch: train={X_train.shape[1]}, test={X_test.shape[1]}"
    
    # Label space consistency
    train_labels = set(np.unique(y_train))
    test_labels = set(np.unique(y_test))
    assert train_labels == test_labels, f"Label space mismatch: train={train_labels}, test={test_labels}"
    
    # Data type consistency
    assert X_train.dtype == X_test.dtype, f"Feature dtype mismatch: train={X_train.dtype}, test={X_test.dtype}"
    assert y_train.dtype == y_test.dtype, f"Label dtype mismatch: train={y_train.dtype}, test={y_test.dtype}"
    
    # Check feature matrices are not empty
    assert X_train.shape[0] > 0, "Training feature matrix is empty"
    assert X_test.shape[0] > 0, "Test feature matrix is empty"
    assert X_train.shape[1] > 0, "No features extracted for training"
    assert X_test.shape[1] > 0, "No features extracted for testing"
    
    # Check total sample conservation
    original_count = len(raw_dataset)
    total_samples = X_train.shape[0] + X_test.shape[0]
    assert total_samples == original_count, f"Sample count mismatch: original={original_count}, splits={total_samples}"
    
    # Check feature values are reasonable (non-negative for CountVectorizer)
    assert (X_train >= 0).all(), "Training features contain negative values"
    assert (X_test >= 0).all(), "Test features contain negative values"
    
    # Check no NaN or infinite values
    assert not np.isnan(X_train).any(), "Training features contain NaN values"
    assert not np.isnan(X_test).any(), "Test features contain NaN values"
    assert not np.isinf(X_train).any(), "Training features contain infinite values"
    assert not np.isinf(X_test).any(), "Test features contain infinite values"




