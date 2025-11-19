# TODO: FeatureCalculator interface
# TODO: Reporter interface

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import datetime

from mltools.feature_store.core import Metadata

class FeatureCalculator(ABC):
    features : list[Metadata]
    
    @abstractmethod
    def _compute(self, **kwargs) -> dict[Metadata, pd.DataFrame]:
        '''Compute the feature based using any arguments necessary.'''
        pass

class FeatureAutomat(ABC):

    @abstractmethod
    def setup(self, config: dict):
        '''
        Set the features to calculate based on a list of tuples containing feature name and module name.
        
        Responsible for setting:
        self.feature_calculators : dict[str, FeatureCalculator]
        '''
        pass

    @abstractmethod
    def compute_universal_kwargs(self, entities_to_calculate : np.ndarray, reference_times: np.ndarray[datetime.datetime]):
        '''Set the features to calculate based on a list of tuples containing feature name and module name.'''
        pass


    # TODO:
    # calculate_features is responsible for defining entities to calculate and is only given a single reference time. This is useful for continuous predictions
    # But for training, we need to typically calculate for small amount of entities (to be passed as argument) and for a large number of reference times (also passed)
    #
    # URGENCY: needed for trainer
    #
    # - what is now "calculate_features" should be a hidden function that is called by the public "calculate_features_for_entities" function + it should receive entities_to_calculate as a parameter
    # - "calculate_features_for_entities" should:
    #     - bucket reference times into time buckets (e.g. 1 day, 1 week, etc.)
    #     - call "calculate_features" for each bucket of reference times with a derived subset of entities
    #     - if no entities are passed it should _fetch_entities_to_calculate instead (i.e. calculate for all entities)
    #     - it should support force_writing of features (i.e. writing even if the values are not changed) - this is of minor importance
    """
    def calculate_features_for_entities(self, entities: list[ID], reference_times: list[datetime.datetime]):
        pass
    """
