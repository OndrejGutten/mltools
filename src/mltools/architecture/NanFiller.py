import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Literal

class NanFiller(BaseEstimator, TransformerMixin):
    def __init__(self, fillna_dict, add_validity_column: bool | dict = False, validity_column_suffix: str = '_is_valid'):
        self.fillna_dict = fillna_dict
        self.add_validity_column = add_validity_column
        self.validity_column_suffix = validity_column_suffix

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_filled = X.copy()

        if self.add_validity_column:
            if isinstance(self.add_validity_column, dict):
                for col, suffix in self.add_validity_column.items():
                    X_filled[f"{col}{suffix}"] = X_filled[col].notna()
            else:
                X_filled[f"{col}{self.validity_column_suffix}"] = X_filled[col].notna()

        for col, fill_value in self.fillna_dict.items():
            if col in X_filled.columns:
                X_filled[col] = X_filled[col].fillna(fill_value)


        return X_filled