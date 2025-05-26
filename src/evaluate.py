"""
This module evaluates a trained model using test data and outputs the evaluation metrics.
"""

import os
import json
import argparse
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def evaluate_model(classifier, X_test, y_test) -> dict:
    """
    Evaluate the model using test data.
    Args:
        classifier: The trained model to evaluate.
        X_test: Test features.
        y_test: Test labels.
    Returns:
        dict: A dictionary containing accuracy and confusion matrix.
    """
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return {
        "accuracy": float(accuracy),
        "confusion_matrix": cm.tolist()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_file", type=str)
    parser.add_argument("X_test", type=str)
    parser.add_argument("y_test", type=str)
    parser.add_argument(
        "--output_filename",
        type=str,
        required=False,
        default="metrics.json"
    )
    parser.add_argument("--output_dir", type=str, required=False, default="metrics")
    args = parser.parse_args()

    loaded_classifier = joblib.load(args.model_file)
    X_test_data = np.load(args.X_test)
    y_test_data = np.load(args.y_test)
    metrics = evaluate_model(loaded_classifier, X_test_data, y_test_data)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, args.output_filename), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
