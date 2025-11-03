import datetime
import numpy as np
import sqlalchemy.engine
from mltools.utils import errors, report, utils as general_utils
from mltools.feature_store.utils import utils
import mltools.feature_store.core.interface as interface
from mltools.feature_store.core import FeatureCollector, FeatureStoreClient

# TODO: remove path_to_db_pickle
class FeatureAutomat(interface.FeatureAutomat):
    def __init__(self,
                 primary_db_engine : sqlalchemy.engine.base.Engine,
                 feature_store_client : FeatureStoreClient,
                 report_name: str,
                 ):
        self.primary_db_engine = primary_db_engine
        self.feature_store_client = feature_store_client

        self.compute_kwargs = {}
        self.feature_calculators = {}

        self.report = report.Report(report_name)

        try:
            self.feature_store_client.connect()
        except Exception as e:
            raise errors.DatabaseConnectionError(f"Failed to connect to databases: {e}")
        
    def calculate_features(self, reference_times: np.ndarray[datetime.datetime], entities_to_calculate: np.ndarray):

        if len(entities_to_calculate) != len(reference_times):
            raise ValueError(f"Length of reference_times {len(reference_times)} must be either 1 (shall be used for all entities_to_calculate) or equal to length of entities_to_calculate {len(entities_to_calculate)} (shall be paired 1-to-1)")
        
        # trigger calculation of features
        self._connect_to_databases()

        self.report.add("Number of candidate entities", len(entities_to_calculate))

        # obtain input data for feature calculation
        # This function needs to be implemented in the subclass
        self.compute_universal_kwargs(entities_to_calculate=entities_to_calculate, reference_times=reference_times) # sets self.compute_kwargs with necessary data for feature calculation

        # compute features
        for feature_calculator in self.feature_calculators.values():
            # fetch prerequisite features
            # TODO: use feature_collector here instead of fetching prerequisite features manually
            prerequisite_features, mask_to_ignore_due_invalid_prerequisite = self._fetch_prerequisite_features(
                feature_calculator = feature_calculator,
                requested_entities= entities_to_calculate,
                reference_times = reference_times)
            # TODO: select rows that are not given in entities_to_ignore_due_to_missing_prerequisite and modify all arguments passed later acoordingly
            entities_to_calculate = prerequisite_features.loc[~mask_to_ignore_due_invalid_prerequisite, :].index.to_numpy()
            reference_times_to_use = reference_times[~mask_to_ignore_due_invalid_prerequisite]
            self.report.add([feature_calculator.address, 'entities_to_calculate'], len(entities_to_calculate))
            self.report.add([feature_calculator.address, 'entities_ignored_due_to_prerequisites'], sum(mask_to_ignore_due_invalid_prerequisite))

            # compute the feature on entities that 1) need recalculation and 2) have all prerequisite features available
            calculated_df = feature_calculator.compute(
                dlznik_ids = general_utils.to_array(entities_to_calculate, dtype = entities_to_calculate.dtype),
                reference_times = reference_times_to_use,
                prerequisite_features = prerequisite_features.loc[~mask_to_ignore_due_invalid_prerequisite, :],
                feature_store_client = self.feature_store_client,
                **self.compute_kwargs,
            )
            self.report.add([feature_calculator.address, 'calculated_rows'], calculated_df.shape[0])

            # split the resulting DataFrame into multiple DataFrames if there is more than one feature in the DataFrame
            # and write each DataFrame to the target database
            single_feature_dfs = utils.split_multifeature(calculated_df, data_columns=feature_calculator.get_feature_names())
            for df, feature_name in zip(single_feature_dfs, feature_calculator.get_feature_names()):
                feature_metadata = next(metadata for metadata in feature_calculator.features if metadata.name == feature_name)
                if feature_metadata.type == interface.FeatureType.EVENT:
                    # write event feature to the target database
                    written_data = self.feature_store_client.write_feature(
                        feature_name=feature_name,
                        module_name=feature_calculator.module_name,
                        feature_df=df,
                        unique_ID_column=feature_calculator.event_id_column
                    )
                elif feature_metadata.type == interface.FeatureType.STATE or feature_metadata.type == interface.FeatureType.TIMESTAMP:
                    # update state feature in the target database
                    # TODO: use write_feature for event features and update_feature for state_features
                    written_data = self.feature_store_client.update_feature(
                        feature_name=feature_name,
                        module_name=feature_calculator.module_name,
                        feature_df=df,
                        value_column=feature_name,
                        reference_time_column='reference_time',
                        groupby_key=self.entity_id_column,
                    )
                self.report.add([feature_calculator.address, feature_name, 'written_rows'], written_data.shape[0])
        
        self._disconnect_from_databases()

    def _disconnect_from_databases(self):
        # disconnect from the databases
        self.feature_store_client.disconnect()

    def _fetch_prerequisite_features(self, feature_calculator , requested_entities : list, reference_times: np.ndarray[datetime.datetime]):
        # fetch prerequisite features for the given feature - only target prerequisite features are currently supported

        '''
        requested_entities = set(requested_entities)
        prerequisite_features = {}
        entities_to_ignore_due_to_missing_prerequisite = set()
        for target_prerequisite_feature in feature.target_prerequisite_features:
            target_prerequisite_df = self.feature_store_connector.load_most_recent_feature_value_wrt_reference_time(
                feature_name=target_prerequisite_feature,
                module_name=feature.module_name,
                reference_time=reference_time,
                groupby_key='dlznik_id',
                reference_time_column='reference_time'
            )
            entities_without_valid_values = utils.entities_with_invalid_attribute(
                                                latest_values_df=target_prerequisite_df,
                                                stale_after_n_days=self.feature_staleness.get(feature.feature_address, None),
                                                id_column='dlznik_id',
                                                reference_time=reference_time
                                            )
            requested_entities_without_valid_values = requested_entities.intersection(entities_without_valid_values)
            print(f""f"Target prerequisite feature '{target_prerequisite_feature}' loaded successfully. Number of requested entities without valid values: {len(requested_entities_without_valid_values)}.")
            entities_to_ignore_due_to_missing_prerequisite = entities_to_ignore_due_to_missing_prerequisite.union(requested_entities_without_valid_values)
            prerequisite_features[target_prerequisite_feature] = target_prerequisite_df[target_prerequisite_df['dlznik_id'].isin(requested_entities.difference(requested_entities_without_valid_values))].copy()
        
        return prerequisite_features, entities_to_ignore_due_to_missing_prerequisite
        '''

        prerequisite_features = {}

        fc = FeatureCollector.FeatureCollector(feature_store_client=self.feature_store_client, path_to_feature_logic=self.path_to_feature_logic)
        collected_features, matched_flags, stale_flags = fc.collect_features(
            entities_to_collect=requested_entities,
            reference_times=general_utils.to_datetime_array(reference_times),
            feature_metadata_address_list=feature_calculator.prerequisite_features,
            id_column=self.entity_id_column,
            reference_time_column='reference_time',
            feature_staleness=self.feature_staleness)
        entities_to_ignore_due_to_missing_prerequisite = ~matched_flags.any(axis=1).to_numpy(bool) if not matched_flags.empty else np.full(len(requested_entities), False)
        entities_to_ignore_due_to_stale_prerequisite = stale_flags.any(axis=1).to_numpy(bool) if not stale_flags.empty else np.full(len(requested_entities), False)
        for feature_metadata_address in feature_calculator.prerequisite_features:
            prerequisite_features[feature_metadata_address] = collected_features[feature_metadata_address]

        return collected_features, entities_to_ignore_due_to_missing_prerequisite | entities_to_ignore_due_to_stale_prerequisite

    def _connect_to_databases(self):
        self.feature_store_client.connect()

