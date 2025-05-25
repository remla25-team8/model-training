import os
import subprocess
import sys

input_file, output_file = sys.argv[1:3]
subprocess.run(["python", "src/train.py", "local-dev"], check=True)
os.rename("model/sentiment_classifier.joblib", output_file)
