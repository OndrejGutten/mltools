import pandas as pd
from mltools.feature_store.utils import utils
from mltools.feature_store.core import interface


def values_not_bound_to_events(
        fs_connector: interface.DB_Connector,
        features_addresses: list[str],
        event_feature_address: str,
        entity_id_column: str,
) -> dict[str, set]:
    # load all events
    # for feature in features:
    # load all values
    # merge events and values on (date, entity_id)
    # any value not having a matching event is considered not bound to an event
    # return a dataframe with unbound values for each feature
    unbound_values = {}
    event_module_name, event_feature_name = utils.address_to_module_and_feature_name(event_feature_address)
    events = fs_connector.load_feature(event_feature_name, event_module_name)
    for feature_address in features_addresses:
        module_name, feature_name = utils.address_to_module_and_feature_name(feature_address)
        feature_values_df = fs_connector.load_feature(feature_name=feature_name, module_name=module_name)
        merged_df = feature_values_df.merge(
            events,
            how = 'left',
            left_on = [entity_id_column, 'reference_time'],
            right_on = [entity_id_column, 'reference_time'],
            indicator = True
        )
        unbound_values[feature_name] = merged_df[merged_df['_merge'] == 'left_only']
    
    return unbound_values


def fs_populated_table(unbound_values :dict):
    reference_time_values_list_of_lists = [list(df['reference_time'].values) for df in unbound_values.values()]
    unique_dates = sorted(set().union(*reference_time_values_list_of_lists))
    table = pd.DataFrame(columns = unbound_values.keys(), index = unique_dates)
    for feature_name, df in unbound_values.items():
        sizes = df.groupby('reference_time').size()
        table[feature_name] = sizes
    return table

def index_time_shift(df: pd.DataFrame):
    df['index'] = df.index
    df['shifted_index'] = df.shift(1)['index']
    df['time_diff'] = df['index'] - df['shifted_index']
    return df