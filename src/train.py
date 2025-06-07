import argparse
import os
import sys
import json
import tempfile
import zipfile
import numpy as np
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from download_model import download_and_load_model
from model_upload import upload_model


def train_model(X_train, y_train, model_version=None, sample_weight=None):
    """
    Train a machine learning model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_version: Version of existing model to load (for transfer learning)
        sample_weight: Sample weights for training
    
    Returns:
        Trained model
    """
    # Check if the model version is provided, in which case we don't train from scratch
    if model_version is None:
        model = GaussianNB()
    else:
        model, _ = download_and_load_model(model_version)

    # Check if the sample weight is provided, in which case we use it to train the model
    if sample_weight is not None:
        if isinstance(sample_weight, str):
            sample_weight = np.load(sample_weight)
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model and return metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
    
    Returns:
        Dictionary containing evaluation metrics
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        "accuracy": float(accuracy),
        "confusion_matrix": cm.tolist()
    }


def save_model_and_metadata(model, metrics, output_dir, model_filename, version=None):
    """
    Save model and metadata to output directory.
    
    Args:
        model: Trained model
        metrics: Evaluation metrics
        output_dir: Output directory
        model_filename: Model filename
        version: Model version
    
    Returns:
        Paths to saved model and metadata files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, model_filename)
    joblib.dump(model, model_path)
    
    # Create metadata
    metadata = {
        "version": version,
        "model_type": type(model).__name__,
        "accuracy": metrics["accuracy"],
        "confusion_matrix": metrics["confusion_matrix"]
    }
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    
    return model_path, metadata_path


def create_model_package(model_path, metadata_path, package_name="model_release"):
    """
    Create a zip package containing model and metadata.
    
    Args:
        model_path: Path to model file
        metadata_path: Path to metadata file
        package_name: Name of the package (without .zip)
    
    Returns:
        Path to created zip file
    """
    zip_path = f"{package_name}.zip"
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(model_path, os.path.basename(model_path))
        zipf.write(metadata_path, os.path.basename(metadata_path))
    
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Train ML model")
    parser.add_argument("mode", choices=["local-dev", "production"], 
                       help="Training mode: local-dev or production")
    parser.add_argument("--version", type=str, required=False, default=None,
                       help="Model version (required for production mode)")
    parser.add_argument("--data-path", type=str, required=False, default=None,
                       help="Path to training data file")
    parser.add_argument("--X-train", type=str, required=False, default="data/splits/X_train.npy",
                       help="Path to training features")
    parser.add_argument("--y-train", type=str, required=False, default="data/splits/y_train.npy",
                       help="Path to training labels")
    parser.add_argument("--X-test", type=str, required=False, default="data/splits/X_test.npy",
                       help="Path to test features")
    parser.add_argument("--y-test", type=str, required=False, default="data/splits/y_test.npy",
                       help="Path to test labels")
    parser.add_argument("--sample-weight", type=str, required=False, default=None,
                       help="Path to sample weights")
    parser.add_argument("--base-model-version", type=str, required=False, default=None,
                       help="Version of existing model to load for transfer learning")
    parser.add_argument("--output-dir", type=str, required=False, default="models",
                       help="Output directory for model")
    parser.add_argument("--model-filename", type=str, required=False, default="sentiment_classifier.joblib",
                       help="Model filename")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == "production" and not args.version:
        print("❌ Error: --version is required for production mode")
        sys.exit(1)
    
    print(f"🚀 Starting {args.mode} training...")
    if args.version:
        print(f"📦 Model version: {args.version}")
    
    # Handle data path for Docker environment
    if args.data_path and os.path.exists(args.data_path):
        print(f"📊 Using data from: {args.data_path}")
        # For Docker environment, we might need to process the data path differently
        # This is a placeholder - you might need to add data loading logic here
    
    # Load training and test data
    try:
        print(f"📁 Loading data from:")
        print(f"  - X_train: {args.X_train}")
        print(f"  - y_train: {args.y_train}")
        print(f"  - X_test: {args.X_test}")
        print(f"  - y_test: {args.y_test}")
        
        X_train = np.load(args.X_train)
        y_train = np.load(args.y_train)
        X_test = np.load(args.X_test)
        y_test = np.load(args.y_test)
        
        print(f"✅ Data loaded successfully")
        print(f"  - Training samples: {len(X_train)}")
        print(f"  - Test samples: {len(X_test)}")
        
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        print("💡 Make sure to run the DVC pipeline first: dvc repro")
        sys.exit(1)
    
    # Train model
    print("🎯 Training model...")
    model = train_model(X_train, y_train, args.base_model_version, args.sample_weight)
    print("✅ Model training completed")
    
    # Evaluate model
    print("📊 Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)
    print(f"📈 Model accuracy: {metrics['accuracy']:.4f}")
    
    # Save model and metadata
    print("💾 Saving model and metadata...")
    model_path, metadata_path = save_model_and_metadata(
        model, metrics, args.output_dir, args.model_filename, args.version
    )
    print(f"✅ Model saved to: {model_path}")
    print(f"✅ Metadata saved to: {metadata_path}")
    
    # Production mode: upload to model registry
    if args.mode == "production":
        print("🚀 Production mode: uploading to model registry...")
        
        # Create model package
        zip_path = create_model_package(model_path, metadata_path)
        print(f"📦 Model package created: {zip_path}")
        
        # Upload to Hugging Face Hub
        try:
            upload_model(zip_path, args.version)
            print(f"🎉 Model version {args.version} uploaded successfully!")
        except Exception as e:
            print(f"❌ Error uploading model: {e}")
            sys.exit(1)
        finally:
            # Clean up zip file
            if os.path.exists(zip_path):
                os.remove(zip_path)
                print(f"🧹 Cleaned up package file: {zip_path}")
    
    else:
        print("🔧 Local development mode: model saved locally only")
    
    print("🎉 Training completed successfully!")


if __name__ == '__main__':
    main()
