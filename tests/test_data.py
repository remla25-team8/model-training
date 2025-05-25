import pytest
import pandas as pd
import numpy as np
from lib_ml.preprocessor import Preprocessor
import os

def test_data_loading():
    """Test that training data loads correctly and has expected columns"""
    dataset = pd.read_csv('data/raw/train_data.tsv', delimiter='\t', quoting=3)
    assert 'Review' in dataset.columns
    assert 'Liked' in dataset.columns
    assert len(dataset) > 0

def test_data_balance():
    """Test that data isn't severely imbalanced"""
    dataset = pd.read_csv('data/raw/train_data.tsv', delimiter='\t', quoting=3)
    class_balance = dataset['Liked'].value_counts(normalize=True)
    assert 0.3 < class_balance[0] < 0.7  # Neither class should dominate

def test_data_quality():
    """Test data quality and integrity"""
    dataset = pd.read_csv('data/raw/train_data.tsv', delimiter='\t', quoting=3)
    
    # Test for missing values
    assert dataset['Review'].isnull().sum() == 0, "Found missing reviews"
    assert dataset['Liked'].isnull().sum() == 0, "Found missing labels"
    
    # Test for empty strings
    assert not (dataset['Review'].str.strip() == '').any(), "Found empty reviews"
    
    # Test data types
    assert dataset['Review'].dtype == object, "Reviews should be strings"
    assert dataset['Liked'].dtype in [int, np.int64], "Labels should be integers"
    
    # Test label values
    assert dataset['Liked'].isin([0, 1]).all(), "Labels should be binary (0 or 1)"

def test_preprocessing_consistency():
    """Test that preprocessing produces consistent results"""
    preprocessor = Preprocessor(max_features=1420)
    sample_text = "This restaurant was great!"
    processed_once = preprocessor.preprocess(sample_text)
    processed_twice = preprocessor.preprocess(sample_text)
    assert processed_once == processed_twice

def test_feature_distribution():
    """Test that feature distribution is reasonable"""
    dataset = pd.read_csv('data/raw/train_data.tsv', delimiter='\t', quoting=3)
    preprocessor = Preprocessor(max_features=1420)
    reviews = dataset['Review']
    preprocessed_reviews = preprocessor.preprocess_batch(reviews)
    X = preprocessor.vectorize(preprocessed_reviews)
    
    # Test feature shape
    assert X.shape[0] == len(dataset)  # Same number of samples
    assert X.shape[1] == 1420  # Expected number of features
    
    # Test feature sparsity - CountVectorizer typically produces very sparse matrices
    sparsity = (X == 0).mean()
    assert sparsity < 0.999, "Features are too sparse (more than 99.9% zeros)"
    
    # Test for constant features
    non_zero_cols = np.any(X != 0, axis=0)
    assert non_zero_cols.sum() > 0, "No non-zero features found"

def test_feature_names():
    """Test that feature names are valid"""
    preprocessor = Preprocessor(max_features=1420)
    sample_texts = ["This is a test review", "Another test review"]
    preprocessed = preprocessor.preprocess_batch(sample_texts)
    X = preprocessor.vectorize(preprocessed)
    
    # Check if vectorizer has feature names
    assert hasattr(preprocessor.vectorizer, 'get_feature_names_out'), "Vectorizer should have feature names"
    feature_names = preprocessor.vectorizer.get_feature_names_out()
    assert len(feature_names) == X.shape[1], "Feature names don't match feature count"