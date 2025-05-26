"""
This script packages the trained sentiment classifier model
and its metrics into a release directory,
and creates a zip archive for distribution.
"""
import os
import shutil

model_dir = "release_model"
os.makedirs(model_dir, exist_ok=True)
shutil.copy(
    os.path.join("models", "sentiment_classifier.joblib"),
    os.path.join(model_dir, "sentiment_classifier.joblib")
)
shutil.copy(
    os.path.join("metrics", "metrics.json"),
    os.path.join(model_dir, "metadata.json")
)
shutil.make_archive("model_release", "zip", model_dir)
