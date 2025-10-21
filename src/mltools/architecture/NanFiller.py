import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class NanFiller(BaseEstimator, TransformerMixin):
    def __init__(self, fillna_dict):
        self.fillna_dict = fillna_dict

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_filled = X.copy()
        for col, fill_value in self.fillna_dict.items():
            if col in X_filled.columns:
                X_filled[col] = X_filled[col].fillna(fill_value)
        return X_filled