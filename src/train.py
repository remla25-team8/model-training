import argparse
from sklearn.naive_bayes import GaussianNB
from download_model import download_and_load_model
import joblib
import os
import numpy as np

def train(X_train, Y_train, model_version = None, sample_weight = None):
    # Check if the model version is provided, in which case we don't train from scratch
    if model_version is None:
        model = GaussianNB()
    else:
        model = download_and_load_model(model_version)
    
    # Check if the sample weight is provided, in which case we use it to train the model
    if sample_weight is not None:
        sample_weight = np.load(sample_weight)
        model.fit(X_train, Y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, Y_train)

    return model
    

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("X_train", type=str)
    args.add_argument("Y_train", type=str)
    args.add_argument("--sample_weight", type=str, required=False, default=None)
    args.add_argument("--model_version", type=str, required=False, default=None)
    args.add_argument("--output_filename", type=str, required=False, default="sentiment_classifier.joblib")
    args.add_argument("--output_dir", type=str, required=False, default="models")
    args = args.parse_args()

    X_train = np.load(args.X_train)
    Y_train = np.load(args.Y_train)

    model = train(X_train, Y_train, args.model_version, args.sample_weight)

    # Create the output directory if it doesn't exist and save the model
    os.makedirs(args.output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(args.output_dir, args.output_filename))
    