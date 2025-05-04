import joblib
import json
import sys
from huggingface_hub import hf_hub_download

def download_and_load_model(version):
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


if __name__ == "__main__":
    # Get version from command line argument if provided
    version = sys.argv[1] 
    
    print(f"Downloading model version: {version}")
    classifier, metadata = download_and_load_model(version)
    print("Successfully downloaded and loaded model!")
    print(f"Model metadata: {metadata}")
    
    



