# Model Training

<!-- BADGES:START -->
![PyLint](https://img.shields.io/badge/PyLint-8.5/10-green)
<!-- BADGES:END -->

This repository will contain the ML training pipeline.
- Library dependencies can be seen in `requirements.txt` and `environment.yaml`.
- Data is preproccessed via our `lib-ml` library.
- The `train.py` script will execute the training steps and make the model accessible publicly via huggingface.

## Running locally
First setup environment using:
```bash
conda env -f environment.yaml
conda activate remla-model-training
```
The relevant files are all in `src` so navigate to that directory in your terminal
```bash
cd src
```

To execute model training without uploading to the model registry run:
```bash
python train.py local-dev
```

To execute training and upload run:
```bash
python train.py production --version <version>
```
Note: You will need to have a unique version. The naming convension will follow v1, v2, v3, etc (For now theres just the version: v1). Additionally you must have a .env with a `HF_TOKEN` set. If you don't have access to the token you must at least have permission to create a version tag which on creation will trigger the model to be trained and uploaded to github.

After uploading you can verify that it worked by running:
```bash
python test_download.py <version>
```


## Running via Docker Container
You can configure build details in the `compose.yaml`. For details on the container environment look at `Dockerfile`.

To run the training via container run the following command:
```bash
docker compose up
```

By default it just runs local-dev version that doesn't upload to registry but this can be changed in the `compose.yaml`

## How to access the resulting model externally.
We use huggingface as the model registy so to use our model you will first need to have the huggingface_hub dependency in your environment. Then you can access the model via a function like:
```python
def download_and_load_model(version="1"):
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
```

## Assignment 4
### 1. DVC setup
Please refer to [dvcREADME.markdown](dvcREADME.markdown)

### 2. ML Testing
```
export PYTHONPATH="$PYTHONPATH:$(pwd)"
ln -s data/raw/train_data.tsv train_data.tsv
pytest tests/ -v  # Run all test files
```

### 3. Linters Checking
Please refer to  [pylint_custom/linter_check.md](pylint_custom/linter_check.md)

### 4. CI/CD Workflow
Please refer to  [workflowREADME.md](workflowREADME.md)
