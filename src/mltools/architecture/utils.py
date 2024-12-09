from sklearn.base import BaseEstimator, TransformerMixin
from abc import ABC, abstractmethod
import mlflow

class BaseModel(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def predict_proba(self, X):
        pass

class PredictTransformerWrapper(BaseEstimator, TransformerMixin):
    """
    A wrapper for sklearn transformers to expose the `predict` method,
    which calls the `transform` method. This allows logging transformers with mlflow as if they were models.
    """
    def __init__(self, transformer, transforms_X = True, transforms_y = False):
        if not (transforms_X ^ transforms_y):
            raise ValueError("Either transforms_X or transforms_y must be True, but not both nor neither.")
        self.transforms_X = transforms_X
        self.transforms_y = transforms_y
        self.transformer = transformer

    def fit(self, X, y=None):
        if self.transforms_X:
            self.transformer.fit(X)
        elif self.transforms_y:
            self.transformer.fit(y)
        return self

    def transform(self, x_or_y):
        return self.transformer.transform(x_or_y)

    def predict(self, x_or_y):
        # Calls the transform method under the hood
        return self.transform(x_or_y)


from enum import Enum
class ModelType(Enum):
    PREDICTOR = 1
    TRANSFORMER = 2

class PyfuncMlflowWrapper(mlflow.pyfunc.PythonModel):
    """
    Wrapper to allow saving models as pyfunc flavor. This allows modifying the outcome of the predict method.
    """
    def __init__(self, model = None, model_type = ModelType.PREDICTOR):
        if model:
            self.model = model
            self.model_type = model_type

    def load_context(self, context):
        pass

    def predict(self, context, model_input, params=None):
        params = params or {"predict_method": "predict"}
        predict_method = params.get("predict_method")

        if predict_method == "predict":
            return self.model.predict(model_input)
        elif predict_method == "predict_proba":
            return self.model.predict_proba(model_input)
        elif predict_method == "predict_log_proba":
            return self.model.predict_log_proba(model_input)
        else:
            raise ValueError(f"The prediction method '{predict_method}' is not supported.")
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def fit(self, X, y):
        self.model.fit(X, y)
