import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# TODO: introduce calibrated classifier to the pipeline


class TF_IDF_Vectorizer_Calibrated_Classifier():
    def __init__(self, classifier: Pipeline, model_name: str):
        self.__repr__ = f"TF_IDF_Vectorizer_Calibrated_Classifier - {model_name}"
        self.name = model_name
        self.clf = Pipeline((
            ('vect', TfidfVectorizer()),
            ('clf', classifier)
        )
        )

    def fit(self, X: pd.DataFrame, y: list | pd.Series):
        self.clf.fit(X, y)
        self.known_labels = np.unique(y)

    def predict(self, input_texts: str):
        return self.clf.predict(input_texts)

    def predict_proba(self, input_texts: str):
            return self.clf.predict_proba(input_texts)