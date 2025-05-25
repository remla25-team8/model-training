import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from src.train import train_model, get_data_splits
from lib_ml.preprocessor import Preprocessor
import logging

def test_non_determinism():
    """Test model is robust to small input variations"""
    # Get trained model and preprocessor
    X_train, X_test, y_train, y_test = get_data_splits()
    classifier, _, _ = train_model()
    preprocessor = Preprocessor(max_features=1420)
    
    # Fit the vectorizer first
    dataset = pd.read_csv('data/raw/train_data.tsv', delimiter='\t', quoting=3)
    reviews = dataset['Review']
    preprocessed_reviews = preprocessor.preprocess_batch(reviews)
    preprocessor.vectorize(preprocessed_reviews)  # This fits the vectorizer
    
    texts = [
        "The food was good",
        "The food was good!",
        "The food was good.",
        "the food was good",
        "The FOOD was GOOD",
        "The food    was     good",  # Extra spaces
        "The food was very good",    # Small word addition
        "The meal was good"          # Synonym
    ]
    
    # Use vectorize_single for each text
    vectors = np.vstack([
        preprocessor.vectorize_single(text) for text in texts
    ])
    predictions = classifier.predict(vectors)
    probabilities = classifier.predict_proba(vectors)
    
    # Test prediction consistency
    assert len(set(predictions)) == 1, "Inconsistent predictions for similar inputs"
    
    # Test probability validity
    assert probabilities.shape == (len(texts), 2), "Wrong probability shape"
    assert np.allclose(np.sum(probabilities, axis=1), 1), "Probabilities don't sum to 1"
    assert np.all((0 <= probabilities) & (probabilities <= 1)), "Invalid probability values"

def test_data_slices():
    """Test model performance on specific data slices"""
    # Get trained model and preprocessor
    X_train, X_test, y_train, y_test = get_data_splits()
    dataset = pd.read_csv('data/raw/train_data.tsv', delimiter='\t', quoting=3)
    classifier, _, _ = train_model()
    preprocessor = Preprocessor(max_features=1420)
    
    # Fit the vectorizer first
    reviews = dataset['Review']
    preprocessed_reviews = preprocessor.preprocess_batch(reviews)
    preprocessor.vectorize(preprocessed_reviews)  # This fits the vectorizer
    
    def evaluate_slice(slice_data, slice_name):
        if len(slice_data) > 0:
            # Use vectorize_single for each review
            vectors = np.vstack([
                preprocessor.vectorize_single(text) for text in slice_data['Review']
            ])
            y = slice_data['Liked']
            acc = accuracy_score(y, classifier.predict(vectors))
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

def test_adversarial_inputs():
    """Test model robustness against adversarial inputs"""
    # Get trained model and preprocessor
    X_train, X_test, y_train, y_test = get_data_splits()
    classifier, _, _ = train_model()
    preprocessor = Preprocessor(max_features=1420)
    
    # Fit the vectorizer first
    dataset = pd.read_csv('data/raw/train_data.tsv', delimiter='\t', quoting=3)
    reviews = dataset['Review']
    preprocessed_reviews = preprocessor.preprocess_batch(reviews)
    preprocessor.vectorize(preprocessed_reviews)  # This fits the vectorizer
    
    adversarial_cases = [
        # Empty or minimal input
        "dummy",  # Replace empty string with dummy text
        "test",   # Replace space with test
        "test.",  # Replace dot with test.
        "test!",  # Replace exclamation with test!
        
        # Extreme length
        "good " * 100,
        "bad " * 100,
        
        # Mixed signals
        "The food was terrible but amazing",
        "Horrible service but the best food ever",
        
        # Sarcasm/Negation
        "Not the worst experience ever",
        "This place is not not not good",
        
        # Special characters
        "Good food",  # Replace emoji text
        "Bad service",  # Replace X text
        "Food review",  # Replace dots
        
        # HTML/Markdown-like text
        "good food",  # Replace HTML
        "excellent service",  # Replace markdown
        
        # Numbers and symbols
        "five stars",  # Replace 5/5
        "one star",    # Replace 1/5
        "Restaurant number one"  # Replace #1
    ]
    
    for text in adversarial_cases:
        try:
            # Should not raise exceptions
            vector = preprocessor.vectorize_single(text)
            prediction = classifier.predict(vector)
            probabilities = classifier.predict_proba(vector)
            
            # Basic validity checks
            assert prediction.shape == (1,), "Invalid prediction shape"
            assert probabilities.shape == (1, 2), "Invalid probability shape"
            assert np.allclose(np.sum(probabilities, axis=1), 1), "Probabilities don't sum to 1"
            
        except Exception as e:
            assert False, f"Failed to handle adversarial input '{text}': {str(e)}"