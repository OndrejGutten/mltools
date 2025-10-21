import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class RawToOHE:
    """
    Preprocessor to convert raw dataframe to numpy array with One-Hot-Encoded categorical features.
    """

    def __init__(self, ohe_features_dict : dict):
        """
        :param ohe_features_dict: Dictionary where keys are feature names in the incoming dataframe and values are lists of categories for One-Hot-Encoding.
        """
        self.ohe = OneHotEncoder(categories=list(ohe_features_dict.values()), sparse_output=False, handle_unknown="error")
        self.ohe_features_dict = ohe_features_dict

    def fit(self, df : pd.DataFrame):
        ohe_features_present = [feature for feature in self.ohe_features_dict.keys() if feature in df.columns]
        self.ohe.fit(df[ohe_features_present])
        return self

    def transform(self, df : pd.DataFrame) -> np.ndarray:
        ohe_features_present = [feature for feature in self.ohe_features_dict.keys() if feature in df.columns]
        ohe_features_transformed = self.ohe.transform(df[ohe_features_present])
        non_transformed_features = df.drop(columns=ohe_features_present, axis=1)
        return np.concatenate((non_transformed_features, ohe_features_transformed), axis=1)
    
    def fit_transform(self, df : pd.DataFrame) -> np.ndarray:
        self.fit(df)
        return self.transform(df)