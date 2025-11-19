import sys
sys.path.insert(0, './src')

import numpy as np
import datetime
import yaml
from mltools.feature_store.core import Automat, Client, interface
from mltools.utils import report

from mltools.feature_store.examples import Calculator

class TestAutomat(Automat.FeatureAutomat):
    def __init__(self, fsc : Client.FeatureStoreClient):
        self.feature_store_client = fsc
        self.report = report.Report('TestAutomat Report')
        self.compute_kwargs = {}


    def setup(self, config: dict):
        self.feature_calculators = config['feature_calculators']

    def compute_universal_kwargs(self, entities_to_calculate : np.ndarray, reference_times: np.ndarray[datetime.datetime]):
        pass


config = {
    'feature_calculators' : {
        'test_feature_calculator' : Calculator.TestFeatureCalculator(),
        'another_test_feature_calculator' : Calculator.AnotherTestFeatureCalculator(),
        'target_feature_calculator' : Calculator.TargetFeatureCalculator()
    }
}

credentials = yaml.safe_load(open('src/mltools/feature_store/examples/DB_credentials.yaml', 'r'))
feature_store_username = credentials['feature_store_username']
feature_store_password = credentials['feature_store_password']
feature_store_address = credentials['feature_store_address']
fsc = Client.FeatureStoreClient(db_flavor = 'postgresql+psycopg2', username=feature_store_username, password=feature_store_password, address=feature_store_address)
fsc.connect()

automat = TestAutomat(fsc)
automat.setup(config)
yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
today = datetime.datetime.now()
calculated_features = automat.calculate_features(
    reference_times = [today] * 6,
    entities_to_calculate = [1, 2, 3, 4, 5, 6]
)

data, submit_report = fsc.submit_features(calculated_features)

print(calculated_features)
print(automat.report.data)
print(submit_report.data)
