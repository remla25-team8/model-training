# Model Training

<!-- BADGES:START -->
![PyLint](https://img.shields.io/badge/PyLint-8.5/10-green)
![Coverage](https://img.shields.io/badge/Coverage-71%-yellow)
![ML%20Test%20Score](https://img.shields.io/badge/ML%20Test%20Score-1.0/7-orange)
<!-- BADGES:END -->

This repository contains the ML training pipeline.
- Library dependencies can be seen in `requirements.txt` and `environment.yaml`.
- Data is preproccessed via our `lib-ml` library.
- The pipeline is executed via dvc which is configured with aws
- The model is made accessible publicly via huggingface.
- Workflows automate the testing and management of this repo.

## Setup
First setup environment using:

```bash
conda env create -f environment.yaml
conda activate remla-model-training
```

You will then need to setup dvc:

```bash
export AWS_ACCESS_KEY_ID='your-access-key-id'
export AWS_SECRET_ACCESS_KEY='your-secret-access-key'
export AWS_DEFAULT_REGION='us-east-1'
dvc pull
```

## The pipeline
In `src/` you can see all the components used in the dvc pipeline.

```
├── src
│   ├── download_data.py
│   ├── download_model.py
│   ├── evaluate.py
│   ├── model_upload.py
│   ├── package_model.py
│   ├── preprocess.py
```

If any modifications are made to these components it's suggested to first run:

```bash
pytest tests/test_infrastructure.py
```

To modify dvc pipeline steps you can look at `dvc.yaml`. 

Then to run the dvc pipeline:
```bash
dvc repro
```

After pipeline executed you can push to remote:
```bash
dvc push
```


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

## CI/CD Workflow
Our ML CI/CD pipeline ensures code quality, model reliability, and automated deployment through a comprehensive testing strategy that follows ML best practices.

### 🔄 Pipeline Triggers

- **Every Push**: Code quality checks (PyLint, Flake8, Bandit) and infrastructure tests
- **Pull Requests**: Full validation pipeline without deployment
- **Version Tags** (`v*`): Complete pipeline including model training, testing, and deployment

### 📋 Pipeline Stages

#### 1. **Code Quality & Linting**
- **PyLint**: Custom ML-specific rules with threshold enforcement
- **Flake8**: Code style and complexity checks
- **Bandit**: Security vulnerability scanning
- **Custom Rules**: Hardcoded path detection, missing random_state validation

#### 2. **Infrastructure Tests (Pre-Pipeline)**
- Validates data pipeline components before training
- Tests data loading, preprocessing, and validation logic
- Ensures environment setup and dependencies work correctly
- **Coverage tracking** for infrastructure components

#### 3. **DVC ML Pipeline**
- Executes the complete machine learning pipeline
- Fails fast if data and features don't pass tests (This is because we test data within the DVC pipeline it).
- Trains model using DVC with S3 remote storage
- Generates model artifacts, metrics, and data splits

#### 4. **Post-Pipeline Testing** (Parallel Execution)
- **Model Quality Tests**: Validates trained model performance and behavior
- **Mutamorphic Tests**: Ensures model consistency under transformations
- **Non-Functional Tests**: Performance, memory usage, and reliability checks
- Each test suite includes **coverage reporting**

#### 5. **Model Deployment** (Tags Only)
- **DVC Push**: Uploads model artifacts to S3 remote storage
- **Hugging Face Upload**: Deploys model to public model registry
- **GitHub Release**: Creates versioned release with model package
- **Download Validation**: Tests model accessibility from Hugging Face

#### 6. **Quality Reporting**
- **ML Test Score**: Comprehensive testing framework score (X/7)
- **Coverage Summary**: Combined test coverage across components
- **Badge Updates**: Automatic README badge updates with latest metrics

### 🧪 Testing Strategy

Our testing follows the **ML Test Score** framework with these categories:

1. **Tests for Features and Data** 


2. **Tests for Model Development**


3. **Tests for ML Infrastructure**

4. **Monitoring Tests** 


### 📊 Quality Gates

- **PyLint Score**: ≥ 6.0/10 (configurable in `.pylintrc`)
- **Test Coverage**: ≥ 60% combined across all test suites
- **ML Test Score**: ≥ 1/7 for passing pipeline
- **Security**: No medium/high severity vulnerabilities




