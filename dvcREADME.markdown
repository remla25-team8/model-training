# REMLA25-Team8 DVC Pipeline and Release Setup

This README documents the setup and troubleshooting steps performed on May 22, 2025, to resolve the `403 Forbidden` error in the DVC pipeline and ensure successful execution of GitHub Actions workflows (`dvc-release.yml` and `release.yml`). These steps enable data versioning, pipeline reproducibility, and GitHub Release publishing for Assignment 4.

## Project Overview
- **Objective**: Implement a machine learning pipeline with DVC for data versioning, reproducible training, and automated model release.
- **S3 Bucket**: `s3://remla25-team8/dvcstore` (region: `us-east-1`).
- **Workflows**:
  - `dvc-release.yml`: Runs DVC pipeline, pushes data to S3, and creates a GitHub Release.
  - `release.yml`: Trains the model in a Docker container and pushes images to GitHub Container Registry.
- **Latest Tag**: `v2.0.7`.

## Prerequisites
To reproduce the setup, ensure the following:
- **AWS Account**:
  - Access to `remla25-team8` S3 bucket in `us-east-1`.
  - IAM user with credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`).
- **GitHub Repository**:
  - Write access to the repository.
  - GitHub Secrets configured (see below).
- **Local Environment**:
  - Python 3.x, DVC (`pip install dvc dvc-s3`), AWS CLI.
  - Docker for building and testing images.

## Setup Instructions
### 1. Configure AWS IAM Permissions
The `403 Forbidden` error was resolved by granting the IAM user full S3 access.

1. Log into AWS Console and navigate to **IAM > Users > [your IAM user]**.
2. Add an inline policy (`S3Remla25Team8Access`):
   ```json
   {
       "Version": "2012-10-17",
       "Statement": [
           {
               "Effect": "Allow",
               "Action": [
                   "s3:GetObject",
                   "s3:PutObject",
                   "s3:ListBucket",
                   "s3:DeleteObject",
                   "s3:HeadObject",
                   "s3:GetBucketLocation"
               ],
               "Resource": [
                   "arn:aws:s3:::remla25-team8",
                   "arn:aws:s3:::remla25-team8/*"
               ]
           }
       ]
   }
   ```
3. Note the IAM user’s ARN (e.g., `arn:aws:iam::123456789012:user/your-username`) for bucket policy setup.

### 2. Configure S3 Bucket
Ensure the `remla25-team8` bucket is accessible.

1. In AWS Console, go to **S3** and verify `remla25-team8` exists in `us-east-1`.
2. Set a bucket policy (**Permissions > Bucket policy**):
   ```json
   {
       "Version": "2012-10-17",
       "Statement": [
           {
               "Effect": "Allow",
               "Principal": {
                   "AWS": "arn:aws:iam::[your-account-id]:user/[your-iam-username]"
               },
               "Action": [
                   "s3:GetObject",
                   "s3:PutObject",
                   "s3:ListBucket",
                   "s3:DeleteObject",
                   "s3:HeadObject",
                   "s3:GetBucketLocation"
               ],
               "Resource": [
                   "arn:aws:s3:::remla25-team8",
                   "arn:aws:s3:::remla25-team8/*"
               ]
           }
       ]
   }
   ```
   - Replace `[your-account-id]` and `[your-iam-username]` with your AWS account ID and IAM username.
3. Ensure **Block public access** is enabled (**Permissions > Block public access**).
4. Verify **Default encryption** uses `SSE-S3` or none (**Properties > Default encryption**).

### 3. Set Up GitHub Secrets
Configure AWS credentials in GitHub.

1. Go to repository **Settings > Secrets and variables > Actions > Repository secrets**.
2. Add or update:
   - `AWS_ACCESS_KEY_ID`: IAM user’s access key ID.
   - `AWS_SECRET_ACCESS_KEY`: IAM user’s secret access key.
   - `HF_TOKEN`: Hugging Face token for model training (if required).
3. Generate new credentials if needed:
   - In AWS Console, go to **IAM > Users > [your user] > Security credentials > Create access key**.
   - Select **Command Line Interface (CLI)** and download credentials.

### 4. Clone and Configure Repository
Set up the project locally.

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/remla25-team8.git
   cd remla25-team8
   ```
2. Install dependencies:
   ```bash
   pip install -r src/requirements.txt
   ```
3. Initialize DVC:
   ```bash
   dvc init --no-scm --force
   dvc remote add -d storage s3://remla25-team8/dvcstore
   ```

### 5. Verify DVC Pipeline
Test data versioning and pipeline execution.

1. Pull data from S3:
   ```bash
   export AWS_ACCESS_KEY_ID='your-access-key-id'
   export AWS_SECRET_ACCESS_KEY='your-secret-access-key'
   export AWS_DEFAULT_REGION='us-east-1'
   dvc pull
   ```
   - Verify `data/processed/train_data_processed.tsv` exists:
     ```bash
     ls data/processed/train_data_processed.tsv
     ```
2. Run the pipeline:
   ```bash
   dvc repro
   dvc push
   ```

### 6. Test GitHub Actions Workflows
Trigger workflows to validate the setup.

1. Push a new tag:
   ```bash
   git tag v2.0.8
   git push origin v2.0.8
   ```
2. Monitor GitHub Actions:
   - Go to **Actions** tab and check `dvc-pipeline` and `train-and-release` jobs.
   - Verify `调试S3访问` step lists `s3://remla25-team8/dvcstore` contents.
   - Ensure `dvc pull` succeeds without `403 Forbidden`.
   - Confirm `train.py` runs in `release.yml` without `FileNotFoundError`.
3. Check GitHub Release:
   - Verify a release is created for `v2.0.8` with `model_release.zip` and Docker images.

## Troubleshooting
If the `403 Forbidden` error persists:
1. **IAM Permissions**:
   - Confirm the IAM policy includes `s3:HeadObject` and `s3:GetBucketLocation`.
   - Check for `Deny` rules in other policies.
2. **S3 Bucket**:
   - Verify `remla25-team8` exists in `us-east-1` and contains `dvcstore/files/md5`.
   - Reapply the bucket policy with the correct IAM user ARN.
3. **GitHub Secrets**:
   - Ensure `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` match the IAM user.
   - Regenerate credentials if expired.
4. **Logs**:
   - Share GitHub Actions logs, especially `调试S3访问` output, with **Member 1**.
5. **DVC Support**:
   - Contact https://dvc.org/support with error details.

## Team Collaboration
- **Member 5**: Verify S3 bucket setup, IAM credentials, and bucket policy.
- **Member 4**: Review workflow changes for compliance.
- **Member 1**: Coordinate DVC pipeline issues and debugging.

## Files
- **Workflows**:
  - `.github/workflows/dvc-release.yml`: Manages DVC pipeline and release.
  - `.github/workflows/release.yml`: Handles model training and Docker image push.
- **DVC Config**:
  - `dvc.yaml`: Defines data pipeline (e.g., preprocessing with `download_data.py`).
  - `dvc.lock`: Tracks data versions.
- **Scripts**:
  - `src/download_data.py`: Preprocesses data.
  - `src/train.py`: Trains the model.
  - `src/package_model.py`: Packages the model for release.

## Reproducing Results
To reproduce the pipeline:
1. Follow the setup instructions above.
2. Run `dvc pull` and `dvc repro` locally to verify data and pipeline.
3. Push a new tag (e.g., `v2.0.8`) to trigger workflows.
4. Check GitHub Actions and Releases for output.

For issues, contact **Member 1** or refer to GitHub Actions logs.