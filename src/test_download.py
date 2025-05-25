import joblib
import json
import sys
from huggingface_hub import hf_hub_download
from typing import Tuple, Dict, Any


def download_and_load_model(version: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Downloads and loads a model and its metadata from Hugging Face Hub.

    Args:
        version: The version/revision of the model to download

    Returns:
        Tuple containing:
            - The loaded classifier model
            - Dictionary containing the model metadata

    Raises:
        Exception: If there are issues downloading or loading the model/metadata
    """
    # Download model and metadata from HF Hub
    model_path = hf_hub_download(
        repo_id="todor-cmd/sentiment-classifier",
        filename="sentiment_classifier.joblib",
        revision=version,
    )

    metadata_path = hf_hub_download(
        repo_id="todor-cmd/sentiment-classifier",
        filename="metadata.json",
        revision=version,
    )

    # Load model and metadata
    classifier = joblib.load(model_path)
    with open(metadata_path) as f:
        metadata = json.load(f)

    return classifier, metadata


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python test_download.py <model_version>")
        sys.exit(1)

    version = sys.argv[1]

    try:
        print(f"Downloading model version: {version}")
        classifier, metadata = download_and_load_model(version)
        print("Successfully downloaded and loaded model!")
        print(f"Model metadata: {metadata}")
    except Exception as e:
        print(f"Error downloading/loading model: {str(e)}")
        sys.exit(1)
