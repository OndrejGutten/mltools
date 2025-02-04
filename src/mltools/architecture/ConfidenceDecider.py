import numpy as np
from mltools.architecture.utils import BaseModel

# model needs to call predict_proba instead of predict
# second model needs to wrap custom logic with a transform/predict interface

class ConfidenceDeciderWrapper(BaseModel):
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        '''
        Produce best-guess predictions and confidence levels for each prediction (High vs Low).

        Parameters
        ----------
        X : array-like
            Input data.

        Returns
        -------
        tuple
            A tuple of two arrays of shape (X.shape[0],). The first array contains the best-guess predictions. The second array contains the confidence levels
        '''
        probas = self.model.predict_proba(X)
        high_confidence = probas.max(axis = 1) > self.threshold
        confidence = ['High' if high else 'Low' for high in high_confidence]
        argmax = probas.argmax(axis = 1)
        return np.array([np.array(self.model.known_labels)[argmax], confidence]).reshape(2,-1).T
    
    def predict_proba(self, X):
        '''
        Call underlying model's predict_proba method.

        Parameters
        ----------
        X : array-like
            Input data.

        Returns
        -------
        array
            The predicted probabilities of the underlying model.
        '''
        return self.model.predict_proba(X)