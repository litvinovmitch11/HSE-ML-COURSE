import nltk
import pandas as pd
import pickle
import re

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


class Model():
    def __init__(self, model_dir=None):
        if model_dir:
            self.model = pickle.load(open(model_dir, 'rb'))
        else:
            self.model = Pipeline(
                [
                    ("preprocess", CountVectorizer()),
                    ("model", LogisticRegression(max_iter=1000)),
                ]
            )

    def train(self, train_dir, test_size=None):
        data = pd.read_csv(train_dir)
        data['text_stemmed'] = data['text'].apply(self._preprocess_sentence_eng)
        if test_size:
            X_train, _, y_train, _ = train_test_split(data.text_stemmed, data.rating, test_size=test_size, random_state=42)
        else:
            X_train, y_train = data.text_stemmed, data.rating
        self.model.fit(X_train, y_train)

    def get_stats(self, test_dir, test_size=None):
        data = pd.read_csv(test_dir)
        data['text_stemmed'] = data['text'].apply(self._preprocess_sentence_eng)
        if test_size:
            _, X_test, _, y_test = train_test_split(data.text_stemmed, data.rating, test_size=test_size, random_state=42)
        else:
            X_test, y_test = data.text_stemmed, data.rating
        pred = self.model.predict(X_test)
        return balanced_accuracy_score(y_test, pred)

    def upload(self, model_dir):
        pickle.dump(self.model, open(model_dir, 'wb'))

    def get_predict(self, data_dir):
        if len(data_dir) > 4 and data_dir[len(data_dir)-4:] == ".csv":
            data = pd.read_csv(data_dir)
            data['text_stemmed'] = data['text'].apply(self._preprocess_sentence_eng)    
            X_test = data.text_stemmed
        else:
            X_test = pd.Series([self._preprocess_sentence_eng(data_dir)])
        return list(self.model.predict(X_test))

    def _preprocess_sentence_eng(self, text):
        stemmer = PorterStemmer()
        out = ' '.join(map(stemmer.stem, re.sub(r"[^\w\s]+", '', text).lower().split()))        
        try:
            stop_words = stopwords.words('english')
        except LookupError:
            nltk.download('stopwords')
            stop_words = stopwords.words('english')
        for word in stop_words:
            out = out.replace(" " + word + " ", " ")
        return out
