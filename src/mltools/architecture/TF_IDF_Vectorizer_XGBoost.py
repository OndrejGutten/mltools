

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# TODO: introduce calibrated classifier to the pipeline


class TF_IDF_Vectorizer_XGBoost():
    def __init__(self, model_name: str, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.3):
        self.__repr__ = f"TF_IDF_Vectorizer_XGB - {model_name}"
        self.name = model_name
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.clf = Pipeline((
            ('vect', TfidfVectorizer()),
            ('clf', XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate))
        )
        )
        self.trained = False

    def fit(self, X: pd.DataFrame, y: list | pd.Series):
        vectorized_texts = self.clf['vect'].fit_transform(X)
        self.clf['clf'].fit(vectorized_texts, y)
        self.trained = True
        self.known_labels = np.unique(y)

    def predict(self, input_texts: str):
        if not self.trained:
            raise Exception("Model not trained")

        return self.clf.predict(input_texts)

    def predict_proba(self, input_texts: str):
            if not self.trained:
                raise Exception("Model not trained")
    
            return self.clf.predict_proba(input_texts)