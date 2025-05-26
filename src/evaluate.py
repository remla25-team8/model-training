import json
import sys

import joblib
import pandas as pd
from lib_ml.preprocessor import Preprocessor
from sklearn.metrics import accuracy_score, confusion_matrix


def evaluate_model(model_file: str, data_file: str, output_file: str) -> None:
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
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    model_file, data_file, output_file = sys.argv[1:4]
    evaluate_model(model_file, data_file, output_file)
