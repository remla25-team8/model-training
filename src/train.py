import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
import argparse
import os
import joblib
from huggingface_hub import HfApi, create_repo
from huggingface_hub import login
import shutil
from lib_ml.preprocessor import Preprocessor

#TODO: This should be removed later and replaced by some secret manager
from dotenv import load_dotenv
load_dotenv()

# TODO: This function will later all by handled by the lib-ml package.
def get_train_data():
    dataset = pd.read_csv('train_data.tsv', delimiter = '\t', quoting = 3)
    preprocessor = Preprocessor()

    reviews = dataset['Review']
    preprocessed_reviews = preprocessor.preprocess_batch(reviews)
    X = preprocessor.vectorize(preprocessed_reviews)
    y = dataset['Liked']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    return X_train, X_test, y_train, y_test

def train_model():
    X_train, X_test, y_train, y_test = get_train_data()

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    acc= accuracy_score(y_test, y_pred)

    return classifier, cm, acc

def upload_model(classifier, cm, acc, version):
    # Login to Hugging Face Hub
    # You need to set your HF_TOKEN environment variable or use login() with your token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("Please set your Hugging Face token as HF_TOKEN environment variable")
    
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
        "version": version
    }
    
    # Save metadata to JSON file
    metadata_path = "model/metadata.json"
    with open(metadata_path, 'w') as f:
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
            print(f"Version {version} already exists. Please use a different version number.")
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
            commit_message=f"Add model version {version}"
        )
        
        
        
        print(f"Model version {version} uploaded successfully to https://huggingface.co/{repo_name}")
    except Exception as e:
        print(f"Error during upload: {e}")
    finally:
        # Clean up by removing the temporary directory
        shutil.rmtree("model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["local-dev", "production"], help="Run mode: local-dev for local training, production for training and uploading to registry")
    parser.add_argument("--version", help="Version number for the model (e.g., 1.0.0)")
    args = parser.parse_args()

    if args.mode == "local-dev":
        print("Running the model training locally without uploading to the model registry")
        classifier, cm, acc = train_model()
        print(f"Model training completed with accuracy: {acc}")
        print(f"Confusion matrix: {cm}")
    elif args.mode == "production":
        print("Running the model training and uploading to the model registry")
        classifier, cm, acc = train_model()
        upload_model(classifier, cm, acc, version=args.version)
    else:
        raise ValueError(f"Invalid mode: {args.mode}, please use local-dev or production")