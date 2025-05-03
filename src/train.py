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

    print(cm)
    print(acc)

    return classifier, cm, acc

def upload_model(classifier, cm, acc):
    pass

if __name__ == "__main__":
    classifier, cm, acc = train_model()
