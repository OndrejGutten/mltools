

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import xgboost


# TODO: introduce calibrated classifier to the pipeline


class TF_IDF_XGBoost():
    def __init__(self, model_name: str, parameters: dict, class_weights: dict = None):
        self.__repr__ = f"TF_IDF_XGB - {model_name}"
        self.name = model_name
        self.parameters = parameters

        self.clf = Pipeline((
            ('vect', TfidfVectorizer()),
            ('clf', xgboost.XGBClassifier(**parameters))
        )
        )
        self.trained = False

    def fit(self, X: pd.DataFrame, y: list | pd.Series):
        self.label_encoder = LabelEncoder()
        transformed_y = self.label_encoder.fit_transform(y)
        vectorized_texts = self.clf['vect'].fit_transform(X)
        #dtrain = xgboost.DMatrix(vectorized_texts, label=transformed_y)
        self.clf['clf'].fit(X = vectorized_texts,y = transformed_y)
        self.trained = True
        self.known_labels = np.unique(y)

    def predict(self, input_texts: str):
        if not self.trained:
            raise Exception("Model not trained")

        return self.label_encoder.inverse_transform(self.clf.predict(input_texts))

    def predict_proba(self, input_texts: str):
            if not self.trained:
                raise Exception("Model not trained")
    
            return self.clf.predict_proba(input_texts)

    def _create_weighted_accuracy_loss_function(class_weights):
        def custom_loss(predictions, dtrain):
            labels = dtrain.get_label().astype(int)
            predictions = predictions.reshape(-1, len(class_weights))
            predictions = np.clip(preds, 1e-7, 1 - 1e-7)  # Avoid numerical issues

            # One-hot encoding of labels
            one_hot = np.zeros_like(predictions)
            one_hot[np.arange(len(labels)), labels] = 1

            # Weighted log loss
            grad = -class_weights[labels][:, None] * (one_hot - predictions)
            hess = class_weights[labels][:, None] * predictions * (1 - predictions)

            return grad.flatten(), hess.flatten()

        return custom_loss