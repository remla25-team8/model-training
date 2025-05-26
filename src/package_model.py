import os
import shutil

model_dir = "release_model"
os.makedirs(model_dir, exist_ok=True)
shutil.copy(
    "models/sentiment_classifier.joblib", f"{model_dir}/sentiment_classifier.joblib"
)
shutil.copy("metrics/metrics.json", f"{model_dir}/metadata.json")
shutil.make_archive("model_release", "zip", model_dir)
