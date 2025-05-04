import pandas as pd
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
import argparse
import os
import joblib
from huggingface_hub import HfApi, create_repo
from huggingface_hub import login
import shutil
#TODO: This should be removed later and replaced by some secret manager
from dotenv import load_dotenv
load_dotenv()

# TODO: This function will later all by handled by the lib-ml package.
def get_train_data():
    dataset = pd.read_csv('train_data.tsv', delimiter = '\t', quoting = 3)

    ps = PorterStemmer()

    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')

    corpus=[]

    for i in range(0, 900):
        review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)

    cv = CountVectorizer(max_features = 1420) 
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

    return X_train, X_test, y_train, y_test

def train_model():
    X_train, X_test, y_train, y_test = get_train_data()

    classifier = GaussianNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    acc= accuracy_score(y_test, y_pred)

    return classifier, cm, acc

def upload_model(classifier, cm, acc):
    # Login to Hugging Face Hub
    # You need to set your HF_TOKEN environment variable or use login() with your token
    if not os.getenv("HF_TOKEN"):
        raise ValueError("Please set your Hugging Face token as HF_TOKEN environment variable")
    
    login(token=os.getenv("HF_TOKEN"))
    
    # Create a temporary directory to save the model
    os.makedirs("model", exist_ok=True)
    
    # Save the model and metadata
    model_path = "model/sentiment_classifier.joblib"
    joblib.dump(classifier, model_path)
    
    # Save model metadata
    metadata = {
        "accuracy": float(acc),
        "confusion_matrix": cm.tolist(),
        "model_type": "GaussianNB",
        "task": "sentiment_analysis"
    }
    
    # Create a new repository on Hugging Face Hub
    # Replace 'your-username' with your actual Hugging Face username
    repo_name = "sentiment-classifier"
    try:
        create_repo(repo_name, repo_type="model", private=False)
    except Exception as e:
        print(f"Repository might already exist: {e}")
    
    # Upload the model and metadata
    api = HfApi()
    api.upload_folder(
        folder_path="model",
        repo_id=repo_name,
        repo_type="model"
    )
    
    # Clean up
    shutil.rmtree("model")
    
    print(f"Model uploaded successfully to https://huggingface.co/{repo_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["local-dev", "production"], help="Run mode: local-dev for local training, production for training and uploading to registry")
    args = parser.parse_args()

    if args.mode == "local-dev":
        print("Running the model training locally without uploading to the model registry")
        classifier, cm, acc = train_model()
        print(f"Model training completed with accuracy: {acc}")
        print(f"Confusion matrix: {cm}")
    elif args.mode == "production":
        print("Running the model training and uploading to the model registry")
        classifier, cm, acc = train_model()
        upload_model(classifier, cm, acc)
    else:
        raise ValueError(f"Invalid mode: {args.mode}, please use local-dev or production")