import mltools
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from mltools.architecture.utils import BaseModel

# TODO: introduce calibrated classifier to the pipeline

# TODO: Check if importing BaseModel like this works
class TF_IDF_Classifier(BaseModel):
    def __init__(self, baseline_classifier: Pipeline, model_name: str, class_weights: dict = None):
        self.__repr__ = f"TF_IDF_Calibrated_Classifier - {model_name}"
        self.baseline_classifier = baseline_classifier
        self.name = model_name
        self.class_weights = class_weights

        if class_weights:
            self.set_class_weights(class_weights)
        
    def set_class_weights(self, class_weights: dict):
        self.class_weights = class_weights
        self.baseline_classifier.class_weight = class_weights

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

    