import pandas as pd
from sklearn.model_selection import train_test_split
from lib_ml.preprocessor import Preprocessor
from typing import Tuple
import numpy as np
import os
import argparse

def preprocess_data(dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    preprocessor = Preprocessor()

    reviews = dataset['Review']
    preprocessed_reviews = preprocessor.preprocess_batch(reviews)
    X = preprocessor.vectorize(preprocessed_reviews)
    y = dataset['Liked']

    return X, y

def get_data_splits(raw_data_path: str, test_size: float = 0.20, random_state: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess training data.
    
    Returns:
        Tuple containing:
            - X_train: Training features 
            - X_test: Test features
            - y_train: Training labels
            - y_test: Test labels
    """
    dataset = pd.read_csv(raw_data_path, delimiter='\t', quoting=3)

    X, y = preprocess_data(dataset)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # Parse command line arguments, these are the params for dvc pipeline to pass.
    args = argparse.ArgumentParser()
    args.add_argument("raw_data_path", type=str)
    args.add_argument("--test_size", type=float, required=False, default=0.20)
    args.add_argument("--random_state", type=int, required=False, default=0)
    args.add_argument("--data_splits_dir", type=str, required=False, default="data/splits")
    
    args = args.parse_args()

    # Get the data splits
    X_train, X_test, y_train, y_test = get_data_splits(args.raw_data_path)

    # Save the data splits to the data splits directory
    os.makedirs(args.data_splits_dir, exist_ok=True)
    
    np.save(os.path.join(args.data_splits_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(args.data_splits_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(args.data_splits_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(args.data_splits_dir, 'y_test.npy'), y_test)


