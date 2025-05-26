"""
This module handles uploading a trained model to the Hugging Face Hub.
"""

import os
import argparse
import zipfile
import shutil
import tempfile
from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo, login

load_dotenv()


def upload_model(model_zip_path: str, version: str) -> None:
    """
    Upload trained model and metadata to Hugging Face Hub.

    Args:
        model_zip_path (str): Path to the trained model zip file.
        version (str): Model version string.

    Raises:
        ValueError: If HF_TOKEN environment variable is not set.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Please set your Hugging Face token as HF_TOKEN environment variable")

    login(token=hf_token)

    # Create a new repository on Hugging Face Hub
    repo_name = "todor-cmd/sentiment-classifier"

    # Upload the model and metadata
    api = HfApi()

    # Create a temporary directory for extraction
    temp_dir = tempfile.mkdtemp()
    try:
        print(f"Extracting {model_zip_path} to temporary directory...")
        with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # Upload the model and metadata
        api = HfApi()

        try:
            # Create repository if it doesn't exist
            if not api.repo_exists(repo_name) or repo_name == "test":
                create_repo(repo_name, repo_type="model", private=False)
            elif api.revision_exists(repo_name, version):
                print(f"Version {version} already exists. Please use a different version number.")
                return
            else:
                api.create_branch(repo_id=repo_name, branch=version, revision="main")

            print(f"Uploading model version {version} to {repo_name}")

            # Upload model folder with the specified version
            api.upload_folder(
                folder_path=temp_dir,
                repo_id=repo_name,
                repo_type="model",
                revision=version,
                commit_message=f"Add model version {version}"
            )

            print(f"Model version {version} uploaded successfully to https://huggingface.co/{repo_name}")
        except Exception as upload_error:
            print(f"Error during upload: {str(upload_error)}")

    finally:
        # Clean up: remove the temporary directory
        print("Cleaning up temporary directory...")
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_zip_path", type=str)
    parser.add_argument("version", type=str)
    args = parser.parse_args()
    upload_model(args.model_zip_path, args.version)
