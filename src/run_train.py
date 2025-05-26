"""
This script serves as an entry point for training a sentiment classifier model.
It expects two command-line arguments: an input file and an output file path.
The script runs the training process by invoking 'src/train.py' with the
'local-dev' argument, and then renames the resulting model file to the
specified output file path.
"""
import os
import subprocess
import sys

input_file, output_file = sys.argv[1:3]
subprocess.run(["python", "src/train.py", "local-dev"], check=True)
os.rename("model/sentiment_classifier.joblib", output_file)
