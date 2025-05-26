"""
This module provides functions to preprocess data and split it into training and test sets.
"""

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lib_ml.preprocessor import Preprocessor


def preprocess_data(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Preprocess the dataset by cleaning and vectorizing the reviews.
    Args:
        dataset (pd.DataFrame): The dataset containing reviews and labels.
    Returns:
        tuple: A tuple containing the features and labels.
    """
    preprocessor = Preprocessor()

    reviews = dataset['Review']
    preprocessed_reviews = preprocessor.preprocess_batch(reviews)
    X = preprocessor.vectorize(preprocessed_reviews)
    y = dataset['Liked']

    return X, y


def get_data_splits(
    raw_data_path: str,
    test_size: float = 0.20,
    random_state: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess training data.
    Returns:
        tuple: A tuple containing training and test features and labels.
    """
    dataset = pd.read_csv(raw_data_path, delimiter='\t', quoting=3)

    X, y = preprocess_data(dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    # Parse command line arguments, these are the params for dvc pipeline to pass.
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_data_path", type=str)
    parser.add_argument("--test_size", type=float, required=False, default=0.20)
    parser.add_argument("--random_state", type=int, required=False, default=0)
    parser.add_argument(
        "--data_splits_dir",
        type=str,
        required=False,
        default="data/splits"
    )
    args = parser.parse_args()

    # Get the data splits
    X_train_split, X_test_split, y_train_split, y_test_split = get_data_splits(args.raw_data_path)

    # Save the data splits to the data splits directory
    os.makedirs(args.data_splits_dir, exist_ok=True)
    np.save(os.path.join(args.data_splits_dir, 'X_train.npy'), X_train_split)
    np.save(os.path.join(args.data_splits_dir, 'X_test.npy'), X_test_split)
    np.save(os.path.join(args.data_splits_dir, 'y_train.npy'), y_train_split)
    np.save(os.path.join(args.data_splits_dir, 'y_test.npy'), y_test_split)
