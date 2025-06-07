import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from train import train_model
from lib_ml.preprocessor import Preprocessor
import logging
import pytest
import joblib
import re
import random


@pytest.fixture
def get_splits():
    X_train = np.load('data/splits/X_train.npy')
    X_test = np.load('data/splits/X_test.npy')
    y_train = np.load('data/splits/y_train.npy')
    y_test = np.load('data/splits/y_test.npy')
    return X_train, X_test, y_train, y_test


@pytest.fixture
def trained_model():
    """Fixture that provides a trained model"""
    # Load trained model
    classifier = joblib.load('models/sentiment_classifier.joblib')
    return classifier


@pytest.fixture
def dataset():
    """Fixture that provides the training dataset"""
    return pd.read_csv('data/raw/raw_data.tsv', delimiter='\t', quoting=3)


@pytest.fixture
def preprocessor(dataset):
    """Fixture that provides a fitted preprocessor"""
    preprocessor = Preprocessor(max_features=1420)
    reviews = dataset['Review']
    preprocessed_reviews = preprocessor.preprocess_batch(reviews)
    preprocessor.vectorize(preprocessed_reviews)  # This fits the vectorizer
    return preprocessor


def test_data_slices(dataset, trained_model, preprocessor):
    """Test model performance on specific data slices"""
    def evaluate_slice(slice_data, slice_name):
        if len(slice_data) > 0:
            # Use vectorize_single for each review
            vectors = np.vstack([
                preprocessor.vectorize_single(text) for text in slice_data['Review']
            ])
            y = slice_data['Liked']
            acc = accuracy_score(y, trained_model.predict(vectors))
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


def test_metamorphic_negation(trained_model, preprocessor, dataset):
    # Skip negation test for very small datasets as it's unreliable with simple models
    if len(dataset) < 100:
        pytest.skip("Negation test unreliable with small datasets and simple models")
    
    def apply_negation(text):
        """Apply simple negation by adding 'not' strategically"""
        # Simple heuristic: add "not" after common auxiliary verbs or "was/is"
        text = re.sub(r'\b(is|was|were|are)\b', r'\1 not', text, count=1)
        if 'not' not in text:
            # If no auxiliary verb found, prepend with "It is not the case that"
            text = "It is not the case that " + text.lower()
        return text

    reviews = dataset['Review']
    labels = dataset['Liked']

    failed_cases = []
    total_cases = len(reviews)

    for review, label in zip(reviews, labels):
        original_vector = preprocessor.vectorize_single(review)
        original_pred = trained_model.predict(original_vector)[0]
        negated_text = apply_negation(review)
        negated_vector = preprocessor.vectorize_single(negated_text)
        negated_pred = trained_model.predict(negated_vector)[0]

        if original_pred == negated_pred:
            failed_cases.append({
                'original_text': review,
                'transformed_text': negated_text,
                'prediction': original_pred,
                'true_label': label
            })

    metamorphic_assertion(total_cases, failed_cases, "negation")


def test_metamorphic_trivial_addition(trained_model, preprocessor, dataset):
    def apply_trivial_addition(text):
        """Add neutral phrases that shouldn't change sentiment"""
        neutral_additions = [
            ", you know,",
            ", to be honest,",
            ", I must say,",
            ", in my opinion,",
            ", generally speaking,"
        ]
        addition = random.choice(neutral_additions)
        # Insert addition randomly in the middle or end
        words = text.split()
        if len(words) > 2:
            insert_pos = random.randint(1, len(words))
            words.insert(insert_pos, addition)
            return " ".join(words)
        else:
            return text + addition

    reviews = dataset['Review']
    labels = dataset['Liked']

    failed_cases = []
    total_cases = len(reviews)

    for review, label in zip(reviews, labels):
        original_vector = preprocessor.vectorize_single(review)
        original_pred = trained_model.predict(original_vector)[0]
        trivial_text = apply_trivial_addition(review)
        trivial_vector = preprocessor.vectorize_single(trivial_text)
        trivial_pred = trained_model.predict(trivial_vector)[0]

        if original_pred != trivial_pred:
            failed_cases.append({
                'original_text': review,
                'transformed_text': trivial_text,
                'prediction': original_pred,
                'true_label': label
            })

    metamorphic_assertion(total_cases, failed_cases, "trivial_addition")


def test_metamorphic_intensification(trained_model, preprocessor, dataset):
    def apply_intensification(text, sentiment_label):
        """Add intensifiers based on sentiment"""
        if sentiment_label == 1:  # Positive
            intensifiers = ["very", "extremely", "really", "absolutely"]
            # Look for positive adjectives to intensify
            text = re.sub(
                r'\b(good|great|excellent|amazing|wonderful|nice|fantastic)\b',
                lambda m: f"{random.choice(intensifiers)} {m.group()}", text, count=1
            )
        else:  # Negative
            intensifiers = ["terribly", "extremely", "really", "absolutely"]
            # Look for negative adjectives to intensify
            text = re.sub(
                r'\b(bad|terrible|awful|horrible|poor|disappointing)\b',
                lambda m: f"{random.choice(intensifiers)} {m.group()}", text, count=1
            )
        return text

    reviews = dataset['Review']
    labels = dataset['Liked']

    failed_cases = []
    total_cases = len(reviews)

    for review, label in zip(reviews, labels):
        original_vector = preprocessor.vectorize_single(review)
        original_pred = trained_model.predict(original_vector)[0]
        intensified_text = apply_intensification(review, label)
        intensified_vector = preprocessor.vectorize_single(intensified_text)
        intensified_pred = trained_model.predict(intensified_vector)[0]

        if original_pred != intensified_pred:
            failed_cases.append({
                'original_text': review,
                'transformed_text': intensified_text,
                'prediction': original_pred,
                'true_label': label
            })

    metamorphic_assertion(total_cases, failed_cases, "intensification")


def metamorphic_assertion(total_cases, failed_cases, relation_type):
    # Report results
    success_rate = (total_cases - len(failed_cases)) / total_cases if total_cases > 0 else 0
    logging.info(f"{relation_type} test: {total_cases - len(failed_cases)}/{total_cases} passed ({success_rate:.2%})")

    # Log first few failures for analysis
    if failed_cases:
        logging.warning(f"Found {len(failed_cases)} {relation_type} failures:")
        for i, case in enumerate(failed_cases[:5]):  # Show first 5 failures
            logging.warning(f"  {i + 1}. '{case['original_text']}' -> '{case['transformed_text']}' (both predicted as {case['prediction']})")
        if len(failed_cases) > 5:
            logging.warning(f"  ... and {len(failed_cases) - 5} more failures")

    # Adjust minimum success rate based on dataset size and test type
    if total_cases < 20:  # Very small test dataset
        if relation_type == "negation":
            min_success_rate = 0.1  # 10% minimum for negation on tiny datasets
        else:
            min_success_rate = 0.2  # 20% minimum for other tests on small datasets
    elif relation_type == "negation":  # Negation is harder with simple models
        min_success_rate = 0.3  # 30% minimum for negation
    else:
        min_success_rate = 0.6  # 60% minimum for other tests
        
    assert success_rate >= min_success_rate, \
        f"{relation_type} test failed: {success_rate:.2%} success rate < {min_success_rate:.0%} threshold. Failed cases: {len(failed_cases)}"
