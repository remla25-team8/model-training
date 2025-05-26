"""
This module provides functionality to 
evaluate a trained sentiment analysis model
using a provided dataset and outputs evaluation metrics as JSON.
"""

import json
import sys

import joblib
import pandas as pd
from lib_ml.preprocessor import Preprocessor
from sklearn.metrics import accuracy_score, confusion_matrix


def evaluate_model(model_file: str, data_file: str, output_file: str) -> None:
    """
    Evaluate a trained model on a given dataset 
    and save metrics to a JSON file.

    Args:
        model_file: Path to the trained model file.
        data_file: Path to the evaluation data file.
        output_file: Path to save the evaluation metrics JSON.
    """
    dataset = pd.read_csv(data_file, delimiter="\t", quoting=3)
    preprocessor = Preprocessor()
    reviews = dataset["Review"]
    X = preprocessor.vectorize(preprocessor.preprocess_batch(reviews))
    y = dataset["Liked"]
    classifier = joblib.load(model_file)
    y_pred = classifier.predict(X)
    accuracy = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    metrics = {"accuracy": float(accuracy), "confusion_matrix": cm.tolist()}
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    model_path, data_path, output_path = sys.argv[1:4]
    evaluate_model(model_path, data_path, output_path)
    