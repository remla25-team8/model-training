import argparse
import json
import os
import shutil

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, login
from lib_ml.preprocessor import Preprocessor
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

load_dotenv()


def get_train_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and preprocess training data.

    Returns:
        Tuple containing:
            - X_train: Training features
            - X_test: Test features
            - y_train: Training labels
            - y_test: Test labels
    """
    dataset = pd.read_csv(
        "data/processed/train_data_processed.tsv", delimiter="\t", quoting=3
    )
    preprocessor = Preprocessor()

    reviews = dataset["Review"]
    preprocessed_reviews = preprocessor.preprocess_batch(reviews)
    X = preprocessor.vectorize(preprocessed_reviews)
    y = dataset["Liked"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=0
    )

    return X_train, X_test, y_train, y_test


def train_model() -> tuple[GaussianNB, np.ndarray, float]:
    """
    Train a Gaussian Naive Bayes classifier.

    Returns:
        Tuple containing:
            - classifier: Trained GaussianNB model
            - confusion_matrix: Model confusion matrix
            - accuracy: Model accuracy score
    """
    X_train, X_test, y_train, y_test = get_train_data()

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    return classifier, cm, acc


def upload_model(
    classifier: GaussianNB, cm: np.ndarray, acc: float, version: str
) -> None:
    """
    Upload trained model and metadata to Hugging Face Hub.

    Args:
        classifier: Trained GaussianNB model
        cm: Confusion matrix
        acc: Model accuracy
        version: Model version string

    Raises:
        ValueError: If HF_TOKEN environment variable is not set
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "Please set your Hugging Face token as HF_TOKEN environment variable"
        )

    login(token=hf_token)

    # Create a temporary directory to save the model
    os.makedirs("model", exist_ok=True)

    # Save the model and metadata
    model_path = "model/sentiment_classifier.joblib"
    joblib.dump(classifier, model_path)

    # Save model metadata
    metadata = {
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "model_type": "GaussianNB",
        "task": "sentiment_analysis",
        "version": version,
    }

    # Save metadata to JSON file
    metadata_path = "model/metadata.json"
    with open(metadata_path, "w", encoding="UTF-8") as f:
        json.dump(metadata, f, indent=4)

    # Create a new repository on Hugging Face Hub
    repo_name = "todor-cmd/sentiment-classifier"

    # Upload the model and metadata
    api = HfApi()

    try:
        # Create repository if it doesn't exist
        if not api.repo_exists(repo_name):
            create_repo(repo_name, repo_type="model", private=False)
        elif api.revision_exists(repo_name, version):
            print(
                f"Version {version} already exists. Please use a different version number."
            )
            return
        else:
            api.create_branch(repo_id=repo_name, branch=version, revision="main")

        print(f"Uploading model version {version} to {repo_name}")

        # Upload model folder with the specified version
        api.upload_folder(
            folder_path="model",
            repo_id=repo_name,
            repo_type="model",
            revision=version,
            commit_message=f"Add model version {version}",
        )

        print(
            f"Model version {version} uploaded successfully to https://huggingface.co/{repo_name}"
        )
    except Exception as e:
        print(f"Error during upload: {str(e)}")
    finally:
        # Clean up by removing the temporary directory
        shutil.rmtree("model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["local-dev", "production"],
        help="Run mode: local-dev for local training, production for training and uploading to registry",
    )
    parser.add_argument("--version", help="Version number for the model (e.g., 1.0.0)")
    args = parser.parse_args()

    if args.mode == "local-dev":
        print(
            "Running the model training locally without uploading to the model registry"
        )
        classifier, cm, acc = train_model()
        print(f"Model training completed with accuracy: {acc}")
        print(f"Confusion matrix: {cm}")
        # Save the model for run_train.py
        os.makedirs("model", exist_ok=True)
        joblib.dump(classifier, "model/sentiment_classifier.joblib")
    elif args.mode == "production":
        print("Running the model training and uploading to the model registry")
        classifier, cm, acc = train_model()
        upload_model(classifier, cm, acc, version=args.version)
    else:
        raise ValueError(
            f"Invalid mode: {args.mode}, please use local-dev or production"
        )
