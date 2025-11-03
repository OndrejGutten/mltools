import pandas as pd
import numpy as np
import datetime
from mltools.feature_store.core import interface
from mltools.feature_store.utils import utils
from mltools.utils import utils as general_utils

class FeatureCollector:
    def __init__(self, feature_store_client: interface.FeatureStoreClient, path_to_feature_logic: list[str]):
        self.feature_store_client = feature_store_client
        self.path_to_feature_logic = path_to_feature_logic

    # TODO: rename collect_recent_features - return only most recent features wrt reference time URGENCY: low, not needed until we use FC for fetching EVENTs (for trainer)
    # TODO: add option to return full dfs vs only the values of the features URGENCY: very low
    # TODO: add collect_features_in_date_range - return all features in a date range URGENCY: low, not needed until we use FC for fetching EVENTs (for trainer)
    # TODO: return reference_time as column
    def collect_features(self,
                         entities_to_collect : list,
                         reference_times : datetime.datetime | list[datetime.datetime],
                         feature_metadata_address_list : list[str],
                         id_column: str,
                         reference_time_column = 'reference_time',
                         feature_staleness : dict = {},
                         return_reference_time_column: bool = False,
                         ) -> pd.DataFrame:
        # connect to the feature store
        try:
            self.feature_store_client.connect()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to the feature store: {e}")

        reference_times = general_utils.to_datetime_array(reference_times)

        if len(reference_times) == 1:
            reference_times = np.full(len(entities_to_collect), reference_times[0], dtype='datetime64[ns]')
        elif len(entities_to_collect) != len(reference_times):
            raise ValueError("When passing multiple reference times, the number of entities to collect must match the number of reference times.")
        
        if return_reference_time_column and reference_time_column in feature_metadata_address_list:
            raise ValueError(f"reference_time_column '{reference_time_column}' coincides with one of the requested features in feature_metadata_address_list. Please rename one of them.")
        
        all_features_df = pd.DataFrame(columns = feature_metadata_address_list, index = entities_to_collect)
        matched_df = pd.DataFrame(columns = feature_metadata_address_list, index = entities_to_collect)
        stale_df = pd.DataFrame(columns = feature_metadata_address_list, index = entities_to_collect)

        for feature_metadata_address in feature_metadata_address_list:
            feature_metadata = utils.getFeatureMetaData(feature_metadata_address, self.path_to_feature_logic)

            # load last computed value of feature wrt to reference time
            # ignore stale values
            #TODO: rename reference_time to reference_times
            loaded_feature = self.feature_store_client.load_feature(feature_name=feature_metadata.name,module_name=feature_metadata.module_name)
            historical_data, matched_flag, stale_flag = self.get_historical_data(all_values_df=loaded_feature,
                                                          entities=entities_to_collect,
                                                          reference_times=reference_times,
                                                          id_column=id_column,
                                                          reference_time_column=reference_time_column,
                                                          expiration_days=feature_staleness.get(feature_metadata_address, None))
            
            # assign to all_features_df
            all_features_df[feature_metadata_address] = historical_data
            matched_df[feature_metadata_address] = matched_flag
            stale_df[feature_metadata_address] = stale_flag

        # return_reference_time_column?
        if return_reference_time_column and reference_time_column not in all_features_df.columns:
            all_features_df[reference_time_column] = reference_times
            
        # return the collected features
        return all_features_df, matched_df, stale_df

    def get_historical_data(self, all_values_df: pd.DataFrame, entities: list, reference_times: list, id_column: str, reference_time_column: str, expiration_days: int = None) -> np.ndarray:
        value_column = all_values_df.columns[1]  # assuming the first column is the value column
        null_value = self.dtype_null_value(all_values_df.dtypes[value_column])
        matched_flag_name = '__matched_flag' if '__matched_flag' not in all_values_df.columns else '___matched_flag'
        all_values_df[matched_flag_name] = True
        
        original_order_index_name = '__original_order_index' if '__original_order_index' not in all_values_df.columns else '__old_index'
        entity_df = pd.DataFrame({id_column: entities, reference_time_column: reference_times, original_order_index_name: list(range(len(entities)))})
        
        all_values_df = all_values_df.sort_values(by=[reference_time_column])
        all_values_df[f'{reference_time_column}_feat'] = all_values_df[reference_time_column]

        sorted_entity_df = entity_df.sort_values(by=[reference_time_column])

        # Perform point-in-time join
        merged = pd.merge_asof(
            sorted_entity_df,
            all_values_df,
            by=id_column,
            on=reference_time_column,
            direction="backward",
            suffixes=("", "_feat"),
        )

        merged.sort_values(by=original_order_index_name, inplace=True)

        matched_flag = merged[matched_flag_name].notnull().to_numpy().astype(bool)

        if expiration_days is not None:
            # Remove values that are too old
            age = (merged[reference_time_column] - merged[f"{reference_time_column}_feat"]).dt.days.to_numpy()
            stale_flag = (age > expiration_days).astype(bool)
            merged.loc[:, value_column] = merged.loc[:, value_column].where(~stale_flag, other=null_value)
        else:
            stale_flag = np.full(merged.shape[0], False, dtype=bool)

        if null_value is not None:
            merged.loc[:, value_column] = merged.loc[:, value_column].fillna(null_value)
        return merged[value_column].to_numpy(), matched_flag, stale_flag

    def dtype_null_value(self, dtype):
        if pd.api.types.is_integer_dtype(dtype):
            return np.nan
        elif pd.api.types.is_float_dtype(dtype):
            return np.nan
        elif pd.api.types.is_string_dtype(dtype):
            return None
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            return pd.NaT
        else:
            return np.nan

    def collect_events_in_date_range(self,
                                        feature_name: str,
                                        date_start: datetime.datetime,
                                        date_end: datetime.datetime,
                                        reference_time_column: str = 'reference_time'
                                        ) -> pd.DataFrame:
        # connect to the feature store
        try:
            self.feature_store_client.connect()
        except Exception as e:
            raise ConnectionError(f"Failed to connect to the feature store: {e}")

        feature_metadata = utils.getFeatureMetaData(feature_name, self.path_to_feature_logic)

        # load feature values in the date range
        loaded_feature = self.feature_store_client.load_feature(
            feature_metadata.name,
            feature_metadata.module_name,
        )

        loaded_feature = loaded_feature[
            (loaded_feature[reference_time_column] >= date_start) &
            (loaded_feature[reference_time_column] <= date_end)
        ]

        return loaded_feature