import sys
sys.path.insert(0, './src')

from mltools.feature_store.core.Client import FeatureStoreClient
feature_store_password = "test"
feature_store_username = "test"
feature_store_address = "192.168.1.2:5431/FeatureStoreTest"


fsc = FeatureStoreClient(db_flavor = 'postgresql+psycopg2', username=feature_store_username, password=feature_store_password, address=feature_store_address)
fsc.connect()


from mltools.feature_store.core import Calculator, Metadata, Type
import pandas as pd
import numpy as np
import random

# ========================
meta = Metadata.Metadata(
    feature_name = "test_feature",
    entity_id_name = "dlznik_id",
    feature_type = Type.FeatureType.STATE,
    data_type = 'float',
    stale_after_n_days = 30,
    description = "This is a test feature",
    version_description = "Initial version",
    version = 1
)

class TestFeatureCalculator(Calculator.FeatureCalculator):
    features = [meta] # expected list of Metadata objects
    compute_args = ['dlznik_ids'] # expected list of argument names for _compute method. These are 

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
    feature_name = "another_test_feature",
    entity_id_name = "dlznik_id",
    feature_type = Type.FeatureType.STATE,
    data_type = 'float',
    stale_after_n_days = 30,
    description = "This is a test feature",
    version_description = "Initial version",
    version = 1
)

class AnotherTestFeatureCalculator(Calculator.FeatureCalculator):
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
    feature_name = "target_feature",
    entity_id_name = "dlznik_id",
    feature_type = Type.FeatureType.STATE,
    data_type = 'float',
    stale_after_n_days = 30,
    description = "This is a test feature",
    version_description = "Initial version",
    version = 1
)
class TargetFeatureCalculator(Calculator.FeatureCalculator):
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


from mltools.feature_store.core import Register

print('registered features:')
print(Register._FEATURE_REGISTER)

print('registered feature calculators:')
print(Register._FEATURE_CALCULATOR_REGISTER)

result = TestFeatureCalculator().compute(
    dlznik_ids = np.array([1.0,2.0,3.0,4.0,5.0]),
    irrelevant_argument = np.array(['10', '20', '30', '40', '50']) # this argument will be ignored
)

fsc.update_feature(result[meta], meta)