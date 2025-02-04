import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

from mltools.architecture.utils import BaseModel

# TODO: introduce calibrated classifier to the pipeline


class TF_IDF_Multilabel_Classifier(BaseModel):
    def __init__(self, baseline_classifier: Pipeline, model_name: str):
        self.__repr__ = f"TF_IDF_Multilabel_Calibrated_Classifier - {model_name}"
        self.baseline_classifier = baseline_classifier
        self.name = model_name
        

    def fit(self, X: pd.DataFrame, y: list | pd.Series):
        baseline_pipeline = Pipeline((
            ('vect', TfidfVectorizer()),
            ('clf', self.baseline_classifier)
        ))
        baseline_pipeline.fit(X, y)
        self.clf = CalibratedClassifierCV(estimator=baseline_pipeline, method='isotonic',cv = 'prefit')
        self.clf.fit(X, y)
        self.known_labels = np.unique(y)

    def predict(self, input_texts: str):
        return self.clf.predict(input_texts)

    def predict_proba(self, input_texts: str):
            return self.clf.predict_proba(input_texts)