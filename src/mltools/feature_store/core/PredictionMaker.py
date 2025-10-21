import datetime
import yaml
from mltools.feature_store.core import DB_Connector, FeatureCollector
from mltools.utils import utils
import os
import mlflow
import mltools
import pandas as pd
import numpy as np

class PredictionMaker:
    def __init__(self, credentials_yaml_path: str, config_yaml_path: str):
        self.credentials_yaml_path = credentials_yaml_path
        self.config_yaml_path = config_yaml_path
    
    def verify_setup(self):
        '''
        Verify that the setup is correct by checking the existence of necessary files and configurations. Predictions may still fail if required features are missing or if models cannot be loaded.

        1) check yaml paths (congig + credentials)
        2) read yamls and check if all necessary info present
        3) check for mlflow
        4) check for DB connections
        5) check for reference time
        6) check for models in mlflow
        '''
        # check for DB connections
        # load credentials
        if not os.path.exists(self.credentials_yaml_path):
            raise FileNotFoundError(f"Credentials file not found at {self.credentials_yaml_path}. Please provide a valid path using the --credentials argument.")
        with open(self.credentials_yaml_path, "r") as cred_file:
            credentials = yaml.safe_load(cred_file)
        feature_store_username = credentials.get("feature_store_username", None)
        feature_store_password = credentials.get("feature_store_password", None)
        feature_store_address = credentials.get("feature_store_address", None)
        if feature_store_username is None or feature_store_password is None or feature_store_address is None:
            raise ValueError("Both 'feature_store_username' and 'feature_store_password' and 'feature_store_address' must be provided in the credentials file.")
        feature_store_connection_string = f"{feature_store_username}:{feature_store_password}@{feature_store_address}"

        # create DB connections
        # NOTE: hardcoded PostgreSQL for now
        self.feature_store_connector = DB_Connector.PostgreSQL_DB_Connector(feature_store_connection_string)
        self.feature_store_connector.connect()

        # check yaml paths
        if not os.path.exists(self.credentials_yaml_path):
            raise FileNotFoundError(f"Credentials file not found at {self.credentials_yaml_path}. Please provide a valid path.")
        if not os.path.exists(self.config_yaml_path):
            raise FileNotFoundError(f"Config file not found at {self.config_yaml_path}. Please provide a valid path.")

        # read yamls and check if all necessary info present
        # interpret config
        self.config = yaml.safe_load(open(self.config_yaml_path, "r"))

        self.path_to_feature_logic = self.config.get("path_to_feature_logic", None)
        if self.path_to_feature_logic is None:
            raise ValueError("Path to modules with FeatureCalculators must be specified in the config file under 'path_to_feature_logic'.")
        if not isinstance(self.path_to_feature_logic, list):
            self.path_to_feature_logic = [self.path_to_feature_logic]

        # check for mlflow
        mlflow_server = self.config.get("mlflow_server", None)
        if mlflow_server is None:
            raise ValueError("MLflow server URL must be provided in the configuration file as \"mlflow_server\".")
        
        mlflow.set_tracking_uri(mlflow_server)
        if not mltools.logging.is_remote_mlflow_server_running():
            raise ConnectionError(f"MLflow server at {mlflow_server} is not running or not reachable. Please check the server status.")

        print("Setup verification completed successfully.")

    def predict(self, entities: list, reference_times: list[datetime.datetime], entities_id_column: str):

        reference_times = utils.to_datetime_array(reference_times)

        if len(reference_times) == 1:
            reference_times = np.full(len(entities), reference_times[0], dtype='datetime64[ns]')
        elif len(entities) != len(reference_times):
            raise ValueError("When passing multiple reference times, the number of entities to collect must match the number of reference times.")

        # construct features list to collect (model to features directory mapping + union of all features)
        all_features = set()
        model_to_features = {}
        model_to_mlflow_uri = {}
        models_config = self.config.get("models", [])
        if not models_config:
            raise ValueError("No models specified in the configuration file under 'models'. Please provide at least one model.")
        
        model_to_prediction_address = {}
        for model_entry in models_config:
            model_uri = model_entry.get("model_uri", None)
            prediction_address = model_entry.get("predicition_address", None)
            model_to_prediction_address[model_uri] = prediction_address
            #featureCollectionReport.add("model_uris", model_uri) 

        # Make a set of features required by all models
        for model_uri in model_to_prediction_address.keys():
            #featureCollectionReport.add(model_uri, {})
            model_to_mlflow_uri[model_uri] = model_uri
            mlflow.models.get_model_info(model_uri=model_uri)  # this will raise an error if the model does not exist
            #featureCollectionReport.add([model_uri,'found_in_mlflow'], True)
            # look for artifact with name features.yaml
            try:
                model_feature_address_path = mltools.model.download_model_artifact(model_uri = model_uri, artifact_name = 'feature_list.yaml')
                model_feature_addresses = yaml.safe_load(open(model_feature_address_path, "r"))
                #featureCollectionReport.add([model_uri,'feature_list_read'], True)
                os.remove(model_feature_address_path)  # clean up downloaded file
            except Exception as e:
                print(f"Error downloading feature_list.yaml for model {model_uri}: {e}")
                #featureCollectionReport.add([model_uri,'feature_list_read'], False)
                continue
            
            all_features.update(model_feature_addresses)
            model_to_features[model_uri] = model_feature_addresses

        # collect features
        feature_collector = FeatureCollector.FeatureCollector(feature_store_connector=self.feature_store_connector, path_to_feature_logic=self.path_to_feature_logic)
        all_features_df, matched_df, stale_df = feature_collector.collect_features(
            entities_to_collect=entities,
            reference_times=reference_times,
            feature_metadata_address_list=list(all_features),
            id_column=entities_id_column,
            return_reference_time_column=True  # required by dates-to-timedeltas preprocessor
        )

        # note which entities were not matched or which values were stale
        nonmatched_values = (~matched_df).any(axis=1).index
        #featureCollectionReport.add("nonmatched_values", nonmatched_values)
        stale_values = stale_df.any(axis=1).index
        #featureCollectionReport.add("stale_values", stale_values)

        # make predictions for entities with valid inputs
        for model_uri, prediction_address in model_to_prediction_address.items():
            module_name, feature_name = mltools.feature_store.utils.address_to_module_and_feature_name(prediction_address)
            columns_required_by_model = model_to_features[model_uri] + ['reference_time'] # TODO: read this step from mlflow as well
            #model_inputs = all_features_df[columns_required_by_model]
            nonmatched_values = (~matched_df).any(axis=1).to_numpy()
            stale_values = stale_df.any(axis=1).to_numpy()
            valid_inputs_mask = ~nonmatched_values & ~stale_values
            # identify rows with missing values
            model_ok_inputs = all_features_df[columns_required_by_model][valid_inputs_mask]
            #featureCollectionReport.add([model_uri,'debtors_with_valid_data'], model_ok_inputs.shape[0])
            #featureCollectionReport.add([model_uri,'debtors_with_missing_data'], model_inputs.shape[0] - model_ok_inputs.shape[0])
            if model_ok_inputs.empty:
                print(f"No valid inputs available for model {model_uri} at reference time {reference_times}. Skipping prediction.")
                continue
            # load model from mlflow
            model = mltools.model.load_model({'model' : {'uri' : model_uri}})
            predictions = model.predict_proba(model_ok_inputs)
            # log predictions to DB
            predictions_df = pd.DataFrame({
                entities_id_column: model_ok_inputs.index,
                'prediction': predictions,
                'model_uri': model_uri,
                'reference_time': reference_times[valid_inputs_mask],
                'calculation_time': datetime.datetime.now()
            })
            self.feature_store_connector.update_feature(feature_name=feature_name, module_name=module_name, feature_df=predictions_df, value_column='prediction', groupby_key=entities_id_column)

    def report(self, report_path: str):
        pass
        #featureCollectionReport.add("reference_time", reference_time)
        #featureCollectionReport.save(f"./Intenzita_Volania/reports/{featureCollectionReport.name}.pkl")
        #print(featureCollectionReport.data)
