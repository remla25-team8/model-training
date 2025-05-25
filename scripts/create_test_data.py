"""
Test Data Creator

Creates minimal test datasets for CI/CD testing purposes.
"""

import json
import os
import pandas as pd
from pathlib import Path

def create_test_datasets():
    """Create minimal test datasets for CI/CD pipeline testing."""
    print("🔨 Creating test datasets for CI/CD pipeline...")
    
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("config", exist_ok=True)
    
    # Sample restaurant reviews for testing (all of these text-examples were AI-generated)
    test_reviews = [
        ("This restaurant is amazing! The food was delicious and service was excellent.", 1),
        ("Great atmosphere and friendly staff. Highly recommend!", 1),
        ("Best dining experience I've had in months. Will definitely come back!", 1),
        ("Fresh ingredients and creative dishes. Loved every bite!", 1),
        ("Outstanding flavors and perfect presentation. Worth every penny!", 1),
        ("The food was cold and service was terrible. Very disappointed.", 0),
        ("Overpriced and bland. Would not recommend to anyone.", 0),
        ("Worst dining experience ever. The staff was rude and place was dirty.", 0),
        ("Waited over an hour for mediocre food. Never going back.", 0),
        ("The portions were tiny and the meal was undercooked. Awful!", 0),
        ("Decent food but nothing special. Average experience overall.", 1),
        ("The restaurant was okay. Food was fine but service could be better.", 1),
        ("Not bad but not great either. Probably won't return.", 0),
        ("It was alright, nothing to write home about.", 1),
        ("Mediocre at best. Expected much more for the price.", 0)
    ]
    
    # datasets
    train_df = pd.DataFrame(test_reviews, columns=['Review', 'Liked'])
    train_df.to_csv("data/raw/train_test.tsv", sep='\t', index=False)
    print(f"✅ Created training dataset: {len(train_df)} samples")
    
    val_df = train_df.sample(n=5, random_state=42)
    val_df.to_csv("data/raw/val_test.tsv", sep='\t', index=False)
    print(f"✅ Created validation dataset: {len(val_df)} samples")
    
    # configs and metrics
    test_config = {
        "data": {"train_file": "data/raw/train_test.tsv", "test_size": 0.2},
        "model": {"type": "GaussianNB", "max_features": 50},
        "evaluation": {"metrics": ["accuracy", "precision", "recall", "f1"]}
    }
    
    with open("config/test_config.json", 'w') as f:
        json.dump(test_config, f, indent=2)
    
    os.makedirs("metrics", exist_ok=True)
    train_metrics = {"accuracy": 0.85, "precision": 0.83, "recall": 0.87, "f1": 0.85}
    with open("metrics/train_metrics.json", 'w') as f:
        json.dump(train_metrics, f, indent=2)
    
    print("🎉 Test data creation completed successfully!")

if __name__ == "__main__":
    create_test_datasets()
