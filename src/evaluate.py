import joblib
import json
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import argparse
import os


def evaluate_model(classifier, X_test, y_test) -> dict:
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return {
        "accuracy": float(accuracy),
        "confusion_matrix": cm.tolist()
    }


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("model_file", type=str)
    args.add_argument("X_test", type=str)
    args.add_argument("y_test", type=str)
    args.add_argument("--output_filename", type=str, required=False, default="metrics.json")
    args.add_argument("--output_dir", type=str, required=False, default="metrics")
    args = args.parse_args()

    classifier = joblib.load(args.model_file)
    X_test = np.load(args.X_test)
    y_test = np.load(args.y_test)
    metrics = evaluate_model(classifier, X_test, y_test)

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, args.output_filename), 'w') as f:
        json.dump(metrics, f, indent=4)
