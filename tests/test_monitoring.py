import numpy as np
from train import train_model
import pandas as pd
import logging
import tempfile
from lib_ml.preprocessor import Preprocessor
import os
import pytest
from evaluate import evaluate_model
from sklearn.metrics import accuracy_score


@pytest.fixture
def get_splits():
    X_train = np.load('data/splits/X_train.npy')
    X_test = np.load('data/splits/X_test.npy')
    y_train = np.load('data/splits/y_train.npy')
    y_test = np.load('data/splits/y_test.npy')
    return X_train, X_test, y_train, y_test

# TODO: I think this is not a monitoring test but a data validation test as it's here
# and a monitoring test should probably be implmented in model-service.


def test_data_drift_detection(get_splits):
    """Test data drift detection capabilities for both features and labels"""
    X_train, X_test, y_train, y_test = get_splits

    # Function to compute basic distribution statistics
    def get_distribution_stats(data):
        return {
            'mean': np.mean(data),
            'std': np.std(data),
            'min': np.min(data),
            'max': np.max(data)
        }

    def check_drift(train_data, test_data, data_type="feature"):
        """Check drift between train and test distributions"""
        train_stats = get_distribution_stats(train_data)
        test_stats = get_distribution_stats(test_data)

        for metric in ['mean', 'std']:
            relative_diff = abs(train_stats[metric] - test_stats[metric]) / (
                train_stats[metric] if train_stats[metric] != 0 else 1
            )
            assert relative_diff < 0.5, (
                f"Significant {data_type} drift detected in {metric}: "
                f"train={train_stats[metric]:.3f}, test={test_stats[metric]:.3f}"
            )

    # Check feature drift
    check_drift(X_train, X_test, "feature")

    # Check label drift
    check_drift(y_train, y_test, "label")
