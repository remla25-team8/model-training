import joblib
import json
import sys
from huggingface_hub import hf_hub_download
from typing import Tuple, Dict, Any
from download_model import download_and_load_model


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
