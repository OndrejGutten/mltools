from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
import datetime
from typing import Literal, List, Optional
from mltools.utils import utils

class DatesToTimedeltasPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        timestamp_columns: Optional[List[str]] = None,
        reference_time_column: str = 'reference_time',
        time_unit: Literal['seconds', 'days', 'years'] = 'years'
    ):
        self.timestamp_columns = timestamp_columns
        self.reference_time_column = reference_time_column
        self.time_unit = time_unit

        if self.time_unit == 'seconds':
            self.change_units = self._seconds_to_seconds
        elif self.time_unit == 'days':
            self.change_units = self._seconds_to_days
        elif self.time_unit == 'years':
            self.change_units = self._seconds_to_years  
        else:
            raise ValueError("Invalid time unit. Choose from 'seconds', 'days', or 'years'.")

    def _seconds_to_seconds(self, timedeltas):
        return timedeltas
    def _seconds_to_days(self, timedeltas):
        return timedeltas / 86400
    def _seconds_to_years(self, timedeltas):
        return timedeltas / (86400 * 365.25)

    def _vectorize_timedelta_total_seconds(self, timedeltas):
        def total_seconds_or_nan(td):
            return td.total_seconds() if pd.notna(td) else np.nan
        return np.vectorize(total_seconds_or_nan)(timedeltas)

    def fit(self, X, y=None):
        return self

    def transform(self, X : pd.DataFrame) -> pd.DataFrame:
        if self.reference_time_column not in X.columns:
            raise ValueError(f"Reference time column '{self.reference_time_column}' not found in input DataFrame.")

        reference_times = utils.to_datetime_array(X[self.reference_time_column].to_numpy())

        df = X.drop(columns=[self.reference_time_column]).copy()
        timestamp_columns = self.timestamp_columns
        if timestamp_columns is None:
            timestamp_columns = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]

        if len(reference_times) == 1 and df.shape[0] != 1:
            reference_times = utils.to_datetime_array([reference_times[0]] * df.shape[0])
        elif len(reference_times) != df.shape[0]:
            raise ValueError("Reference timestamps must be either a single datetime or a list of the same length as the column.")

        for col in timestamp_columns:
            dates = utils.to_datetime_array(df[col])
            timedeltas = reference_times - dates
            timedeltas = self._vectorize_timedelta_total_seconds(timedeltas)
            timedeltas = self.change_units(timedeltas)
            df[col] = timedeltas
        return df