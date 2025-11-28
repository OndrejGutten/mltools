import numpy as np

from mltools.feature_store.core import Register, Type

class Metadata:
    name: str
    feature_name : str # unique feature name
    entity_id_name : str # This is a human-readable string identifying the type of entity this features is associated with; 
    feature_type : Type.FeatureType
    data_type = np.dtype
    version : int
    stale_after_n_days : int
    description : str
    version_description : str
    metadata_type = Type.MetadataType.FEATURE

    def __init__(self,
                 feature_name : str,
                 entity_id_name : str,
                 feature_type : Type.FeatureType,
                 data_type : np.dtype,
                 stale_after_n_days : int,
                 description : str,
                 version_description : str,
                 version : int,
                 event_id_name : str = None,
                 value_column = 'value',
                 reference_time_column = 'reference_time'
                ):
        self.feature_name = feature_name
        self.entity_id_name = entity_id_name
        self.feature_type = feature_type
        self.data_type = data_type
        self.stale_after_n_days = stale_after_n_days
        self.description = description
        self.version_description = version_description
        self.version = version
        self.event_id_name = event_id_name
        self.value_column = value_column
        self.reference_time_column = reference_time_column

        Register._FEATURE_REGISTER[self.feature_name] = self

    def __hash__(self):
        return hash(self.feature_name)

    def __eq__(self, other):
        return isinstance(other, Metadata) and self.feature_name == other.feature_name