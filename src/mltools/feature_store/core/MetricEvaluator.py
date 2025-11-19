import mltools
import yaml
import os
import pandas as pd
from mltools.feature_store.core import  FeatureStoreClient, FeatureRegister
from mltools.utils import report
import argparse

class MetricEvaluator():
    def __init__(self, credentials_yaml_path: str, config_yaml_path: str, report_name : str):
        self.credentials_yaml_path = credentials_yaml_path
        self.config_yaml_path = config_yaml_path
        self.report = report.Report(report_name)

    def verify_setup(self):
        '''
        Verify that the setup is correct by checking the existence of necessary files and configurations. Evaluation may still fail if required features are missing or if models cannot be loaded.
        '''
        # check yaml paths
        if not os.path.exists(self.credentials_yaml_path):
            raise FileNotFoundError(f"Credentials file not found at {self.credentials_yaml_path}. Please provide a valid path.")
        if not os.path.exists(self.config_yaml_path):
            raise FileNotFoundError(f"Config file not found at {self.config_yaml_path}. Please provide a valid path.")

        # interpret configs
        self.config = yaml.safe_load(open(self.config_yaml_path, "r"))

        self.model_configs = self.config.get("models", [])
        if not self.model_configs:
            raise ValueError("No model configurations found in the config file.")

        self.path_to_feature_logic = self.config.get("path_to_feature_logic", None)
        if self.path_to_feature_logic is None:
            raise ValueError("Path to modules with FeatureCalculators must be specified in the config file under 'path_to_feature_logic'.")
        if not isinstance(self.path_to_feature_logic, list):
            self.path_to_feature_logic = [self.path_to_feature_logic]

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
        
        # Establish database connections
        feature_store_path = f"{feature_store_username}:{feature_store_password}@{feature_store_address}"
        self.feature_store_client = FeatureStoreClient.FeatureStoreClient(feature_store_path)
        self.feature_store_client.connect()

        print("Setup verification completed successfully.")

    def evaluate(self, range_start, range_end):
        if range_start >= range_end:
            raise ValueError("The start date must be earlier than the end date.")

        for model_config in self.model_configs:
            # staleness limits
            prediction_stale_after_n_days = model_config.get("prediction_stale_after_n_days", None)

            # get ground truth
            ground_truth_address = model_config.get("ground_truth_address","")
            ground_truth_module, ground_truth_feature = mltools.feature_store.utils.address_to_module_and_feature_name(ground_truth_address)
            ground_truth = self.feature_store_client.load_feature(
                                                        ground_truth_feature,
                                                        ground_truth_module,
                                                        timestamp_columns=['reference_time', 'calculation_time'])
            ground_truth = ground_truth[(ground_truth['reference_time'] >= range_start) & (ground_truth['reference_time'] <= range_end)]
            ground_truth = ground_truth.sort_values(by='reference_time').reset_index(drop=True)

            # get predictions
            predictions_address = model_config.get("predictions_address", "")
            predictions_module, predictions_feature = mltools.feature_store.utils.address_to_module_and_feature_name(predictions_address)
            predictions = self.feature_store_client.load_feature(
                                                        predictions_feature,
                                                        predictions_module,
                                                        timestamp_columns=['reference_time', 'calculation_time'])
            predictions = predictions[predictions['model_uri'] == model_config.get("uri")]
            predictions = predictions.sort_values(by='reference_time').reset_index(drop=True)

            id_column = predictions.columns[0]

            most_recent_predictions, matched_flag, stale_flag = self.feature_store_client.get_historical_data(
                all_values_df=predictions,
                entities=ground_truth.iloc[:, 0].to_numpy(),
                reference_times=ground_truth['reference_time'].to_numpy(),
                id_column=id_column,
                reference_time_column='reference_time',
                expiration_days=prediction_stale_after_n_days)


            number_of_missing_predictions = sum(~matched_flag)
            number_of_stale_predictions = sum(stale_flag)
            valid_predictions_flag = matched_flag & ~stale_flag
            number_of_valid_predictions = sum(valid_predictions_flag)
            
            for metric in model_config.get("metrics", []):
                metric_definition = FeatureRegister._FEATURE_CALCULATOR_REGISTER.get(metric, None)
                if metric_definition is None:
                    print(f"Metric calculator '{metric}' not found for model '{model_config.get('uri')}'. Skipping this metric.")
                    continue
                y_true = ground_truth.loc[valid_predictions_flag, ground_truth_feature].to_numpy()
                y_score = most_recent_predictions[valid_predictions_flag]

                if len(y_true) == 0:
                    print(f"No valid predictions for model '{model_config.get('uri')}' in the date range {range_start} - {range_end}. Skipping calculation of {metric_definition.calculator_name}.")
                    self.report.add([model_config.get("uri"), metric_definition.calculator_name,'number_of_observations'], number_of_valid_predictions)
                    self.report.add([model_config.get("uri"), metric_definition.calculator_name,'date_start'], range_start)
                    self.report.add([model_config.get("uri"), metric_definition.calculator_name,'date_end'], range_end)
                    self.report.add([model_config.get("uri"), metric_definition.calculator_name,'kpi_value'], 'NaN')
                    self.report.add([model_config.get("uri"), metric_definition.calculator_name,'number_of_stale_predictions'], number_of_stale_predictions)
                    self.report.add([model_config.get("uri"), metric_definition.calculator_name,'number_of_missing_predictions'], number_of_missing_predictions)
                    continue
                
                kpi_value = metric_definition.compute(
                    y_true = y_true,
                    y_score = y_score
                )

                kpi_measured = pd.DataFrame({
                    'model_uri': [model_config.get("uri")],
                    'value': kpi_value,
                    'start_date': range_start,
                    'end_date': range_end,
                    'calculation_time': pd.Timestamp.now(),
                })

                self.feature_store_client.write_feature(
                    feature_name=metric_definition.calculator_name,
                    module_name=metric_definition.module_name,
                    feature_df=kpi_measured,
                )

                print(f"KPI '{metric_definition.calculator_name}' for model '{model_config.get('uri')}' in the date range {range_start} - {range_end} calculated using {number_of_valid_predictions} values. Number of stale predictions: {number_of_stale_predictions}, Number of missing predictions: {number_of_missing_predictions}. Value: {kpi_value}")
                self.report.add([model_config.get("uri"), metric_definition.calculator_name,'number_of_observations'], number_of_valid_predictions)
                self.report.add([model_config.get("uri"), metric_definition.calculator_name,'date_start'], range_start)
                self.report.add([model_config.get("uri"), metric_definition.calculator_name,'date_end'], range_end)
                self.report.add([model_config.get("uri"), metric_definition.calculator_name,'kpi_value'], kpi_value)
                self.report.add([model_config.get("uri"), metric_definition.calculator_name,'number_of_stale_predictions'], number_of_stale_predictions)
                self.report.add([model_config.get("uri"), metric_definition.calculator_name,'number_of_missing_predictions'], number_of_missing_predictions)

    def report(self):
        pass