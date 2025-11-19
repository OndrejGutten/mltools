import sys
sys.path.insert(0, './src')

import datetime
import yaml
from mltools.feature_store.core import FeatureStoreClient


model_features = ['test_feature', 'another_test_feature']
target_feature_name = ['target_feature']

credentials = yaml.safe_load(open('src/mltools/feature_store/examples/DB_credentials.yaml', 'r'))
feature_store_username = credentials['feature_store_username']
feature_store_password = credentials['feature_store_password']
feature_store_address = credentials['feature_store_address']
fsc = FeatureStoreClient.FeatureStoreClient(db_flavor = 'postgresql+psycopg2', username=feature_store_username, password=feature_store_password, address=feature_store_address)
fsc.connect()

entity_ids = [1,2,3]
reference_times = [datetime.datetime.now() + datetime.timedelta(days=1)] * 3

X_df, _, _ = fsc.collect_features(entities_to_collect=entity_ids,
                            reference_times=reference_times,
                            features_to_collect=model_features)

y_df, _, _ = fsc.collect_features(entities_to_collect=entity_ids,
                            reference_times=reference_times,
                            features_to_collect=target_feature_name)


from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X=X_df.to_numpy(), y=y_df['target_feature'].to_numpy().ravel())

print(X_df)
print(y_df)

prediction = lr.predict(X_df.to_numpy())

print(prediction)
print(y_df['target_feature'].to_numpy())
