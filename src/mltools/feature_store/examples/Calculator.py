import sys
sys.path.insert(0, './src')

from mltools.feature_store.core.FeatureStoreClient import FeatureStoreClient
feature_store_password = "ondrejgutten"
feature_store_username = "ondrejgutten"
feature_store_address = "localhost:5432/pokus"


fsc = FeatureStoreClient(db_flavor = 'postgresql+psycopg2', username=feature_store_username, password=feature_store_password, address=feature_store_address)
fsc.connect()


from mltools.feature_store.core import Metadata, FeatureCalculator
import pandas as pd
import numpy as np
import random

# ========================
meta = Metadata.Metadata(
    name = "test_feature",
    entity_id_name = "dlznik_id",
    feature_type = Metadata.Type.FeatureType.STATE,
    data_type = 'float64',
    stale_after_n_days = 30,
    description = "This is a test feature",
    version_description = "Initial version",
    version = 1
)

class TestFeatureCalculator(FeatureCalculator.FeatureCalculator):
    features = [meta]
    compute_args = ['dlznik_ids']

    def _compute(self, dlznik_ids: np.ndarray) -> pd.DataFrame:
        midnight = pd.Timestamp.now().normalize()
        result_df = pd.DataFrame({
            'entity_id': dlznik_ids,
            'value': 42.0,
            'reference_time': midnight,
            'calculation_time': pd.Timestamp.now(),
            'version': meta.version
        })
        return {meta : result_df}

# ========================

another_meta = Metadata.Metadata(
    name = "another_test_feature",
    entity_id_name = "dlznik_id",
    feature_type = Metadata.Type.FeatureType.STATE,
    data_type = 'float64',
    stale_after_n_days = 30,
    description = "This is a test feature",
    version_description = "Initial version",
    version = 1
)

class AnotherTestFeatureCalculator(FeatureCalculator.FeatureCalculator):
    features = [another_meta]
    compute_args = ['dlznik_ids']

    def _compute(self, dlznik_ids: np.ndarray) -> pd.DataFrame:
        midnight = pd.Timestamp.now().normalize()
        result_df = pd.DataFrame({
            'entity_id': dlznik_ids,
            'value': [random.random() for _ in range(len(dlznik_ids))],
            'reference_time': midnight,
            'calculation_time': pd.Timestamp.now(),
            'version': meta.version
        })
        return {another_meta : result_df}



# ========================
target_meta = Metadata.Metadata(
    name = "target_feature",
    entity_id_name = "dlznik_id",
    feature_type = Metadata.Type.FeatureType.STATE,
    data_type = 'float64',
    stale_after_n_days = 30,
    description = "This is a test feature",
    version_description = "Initial version",
    version = 1
)
class TargetFeatureCalculator(FeatureCalculator.FeatureCalculator):
    features = [target_meta]
    compute_args = ['dlznik_ids']

    def _compute(self, dlznik_ids: np.ndarray) -> pd.DataFrame:
        midnight = pd.Timestamp.now().normalize()
        result_df = pd.DataFrame({
            'entity_id': dlznik_ids,
            'value': [random.random() for _ in range(len(dlznik_ids))],
            'reference_time': midnight,
            'calculation_time': pd.Timestamp.now(),
            'version': meta.version
        })
        return {target_meta : result_df}


from mltools.feature_store.core import FeatureRegister

print('registered features:')
print(FeatureRegister._FEATURE_REGISTER)

print('registered feature calculators:')
print(FeatureRegister._FEATURE_CALCULATOR_REGISTER)

result = TestFeatureCalculator()._compute(np.array([1,2,3,4,5]))

fsc.write_feature(result[meta], meta)