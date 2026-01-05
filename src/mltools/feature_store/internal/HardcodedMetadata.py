
import sys
sys.path.insert(0, ".")
sys.path.insert(0, "/Users/ondrejgutten/Work/PISI.nosync/mltools/src")

from mltools.feature_store.core import Metadata, Register, Type

#TODO: REMOVE - this was necessary for unknown reasons; registering predictions metadata is not thought out properly yet
predictions_dlznik_id_float_metadata = Metadata.Metadata(
    feature_name='dlznik_float',
    entity_id_name='dlznik_id',
    feature_type=Type.FeatureType.STATE,
    data_type='float',
    stale_after_n_days=30,
    description='Predictions for models with entity=dlznik and data_type=float',
    version_description='',
    version = 1,
    value_column='prediction'
)
predictions_dlznik_id_float_metadata.metadata_type = Type.MetadataType.PREDICTION
