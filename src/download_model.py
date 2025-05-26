import joblib
import json
from typing import Any, Dict, Tuple
from huggingface_hub import hf_hub_download


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
        revision=version
    )

    metadata_path = hf_hub_download(
        repo_id="todor-cmd/sentiment-classifier",
        filename="metadata.json",
        revision=version
    )

    # Load model and metadata
    classifier = joblib.load(model_path)
    with open(metadata_path) as f:
        metadata = json.load(f)

    return classifier, metadata
