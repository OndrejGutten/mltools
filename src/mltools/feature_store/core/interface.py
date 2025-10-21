# TODO: DB_Connector interface
# TODO: FeatureCalculator interface
# TODO: Reporter interface

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import datetime
import enum

class FeatureType(enum.Enum):
    # EVENT-type feature represents an event, e.g. a payment, phone call, etc. Each requires an event_id_column attribute with the name of the column that will serve as the unique identifier for the event.
    # Output shall include the event_id column. event_id_column must not be included in attribute_names - this is not a separate attribute, it will be part of the output DataFrame together with the actual value, reference_time, entity_id, calculation_time.
    # Each new event is expected to be added to the database and all events in a date range are expected to be read from the database.
    EVENT = "EVENT" 

    # STATE-type feature represents a state of the entity, e.g. current debt, current number of calls, etc. No further attributes are required.
    # Each new state is expected to be added to the database. Only the most recent state (with respect to a given reference time) is expected to be read from the database.
    STATE = "STATE"

    # HELPER-type feature is used to derive other features, e.g. fetching a list of entities or kwargs to be passed to other FeatureDefinition functions
    # It is not expected to be written to the database, hence its structure is less strict.
    HELPER = "HELPER"

    # TIMESTAMP-type feature is used to store a date instead of timediff. Timediff will be calcuated upon retrieval => will save a lot of storage space.
    # Similar to state - each new state is expected to be added to the database and only the most recent state (with respect to a given reference time) is expected to be read from the database.
    # This type means the value will be a timestamp and should trigger calculation of a timediff (between the timestamp and the reference time) upon retrieval.
    TIMESTAMP = "TIMESTAMP"


class FeatureMetaData:
    name: str
    address: str
    module: str
    feature_type : FeatureType

    def __init__(self, address: str, type: FeatureType):
        self.address = address
        try:
            module_name, name  = address.split('.')
        except ValueError:
            raise ValueError(f"Invalid address format: {address}. Expected format is 'module.name'.")
        self.name = name
        self.module_name = module_name
        self.type = type

# TODO: add feature_calculator 
# TODO: remove FeatureDefinition
class FeatureDefinition(ABC):
    type : FeatureType
    @abstractmethod
    def compute(self, **kwargs) -> pd.DataFrame:
        '''Compute the feature based using any arguments necessary.'''
        pass

    @abstractmethod
    def get_feature_names(self) -> list[str]:
        '''Return the names of the attributes that will be computed by this feature. Each n'''
        pass

class FeatureAutomat(ABC):
    ID : type

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


class DB_Connector(ABC):
    @abstractmethod
    def connect():
        pass

    @abstractmethod
    def disconnect():
        pass

    @abstractmethod
    def load_feature(self, feature_name: str, module_name: str, timestamp_columns: list[str] = None):
        pass

    @abstractmethod
    def load_most_recent_feature_value_wrt_reference_time(self, feature_name: str, module_name : str, reference_time: datetime.datetime,  groupby_key : str, reference_time_column : str):
        pass

    @abstractmethod
    def delete_data(self, feature_name: str, period_start : datetime.datetime, period_end : datetime.datetime, reference_column : str):
        pass
    
    @abstractmethod
    def write_feature(self, feature_name: str, module_name: str, feature_df: pd.DataFrame, unique_ID_column: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def update_feature(self, feature_name: str, module_name: str, feature_df: pd.DataFrame, value_column : str, reference_time_column: str, groupby_key: str) -> pd.DataFrame:
        pass

