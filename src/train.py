"""
This script handles the end-to-end process of training a sentiment
analysis model using a Gaussian Naive Bayes classifier. It loads and
preprocesses data, trains the model, evaluates its performance, and
optionally uploads the trained model and its metadata to the Hugging
Face Hub.

The script supports two modes: 'local-dev' for local training and
saving, and 'production' for training and uploading the model to a
remote registry. Command-line arguments control the mode and model
versioning.
"""
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
    """
    X_train, X_test, y_train, y_test = get_train_data()

    model = GaussianNB()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    test_confusion_matrix = confusion_matrix(y_test, y_pred)
    test_accuracy = accuracy_score(y_test, y_pred)

    return model, test_confusion_matrix, test_accuracy


def upload_model(
    model_to_upload: GaussianNB,
    confusion_matrix_val: np.ndarray,
    accuracy_val: float,
    version: str
) -> None:
    """
    Upload trained model and metadata to Hugging Face Hub.

    Args:
        model_to_upload: Trained GaussianNB model
        confusion_matrix_val: Confusion matrix
        accuracy_val: Model accuracy
        version: Model version string

    Raises:
        ValueError: If HF_TOKEN environment variable is not set
    """
    if not (hf_token := os.getenv("HF_TOKEN")):
        raise ValueError(
            "Please set your Hugging Face token as HF_TOKEN environment variable"
        )

    login(token=hf_token)

    # Create a temporary directory to save the model
    os.makedirs("model", exist_ok=True)

    # Save the model and metadata
    model_path = "model/sentiment_classifier.joblib"
    joblib.dump(model_to_upload, model_path)

    # Save model metadata
    metadata = {
        "accuracy": float(accuracy_val),
        "confusion_matrix": confusion_matrix_val.tolist(),
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
                f"Version {version} already exists. "
                "Please use a different version number."
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
            f"Model version {version} uploaded successfully "
            f"to https://huggingface.co/{repo_name}"
        )
    except (OSError, ValueError, RuntimeError) as err:
        print(f"Error during upload: {str(err)}")
    finally:
        # Clean up by removing the temporary directory
        shutil.rmtree("model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["local-dev", "production"],
        help="Run mode: local-dev for local training, \
            production for training and uploading to registry",
    )
    parser.add_argument("--version", help="Version number for the model (e.g., 1.0.0)")
    args = parser.parse_args()

    if args.mode == "local-dev":
        print(
            "Running the model training locally without uploading to the model registry"
        )
        trained_model_local, confusion_local, accuracy_local = train_model()
        print(f"Model training completed with accuracy: {accuracy_local}")
        print(f"Confusion matrix: {confusion_local}")
        # Save the model for run_train.py
        os.makedirs("model", exist_ok=True)
        joblib.dump(trained_model_local, "model/sentiment_classifier.joblib")
    elif args.mode == "production":
        print("Running the model training and uploading to the model registry")
        trained_model_prod, confusion_prod, accuracy_prod = train_model()
        upload_model(
            trained_model_prod,
            confusion_prod,
            accuracy_prod,
            version=args.version
        )
    else:
        raise ValueError(
            f"Invalid mode: {args.mode}, please use local-dev or production"
        )
