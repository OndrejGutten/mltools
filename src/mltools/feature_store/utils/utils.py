import multiprocessing
import importlib
import datetime
import os
import warnings
import sys
import inspect
import dateutil
from functools import partial
from typing import Callable
import sys
import pandas as pd
import numpy as np
import unicodedata

from mltools.feature_store.core import interface, Metadata
from mltools.utils import errors
from mltools.utils import utils as general_utils

#TODO: staleness must be evaluated wrt multiple reference times
def entities_with_invalid_attribute(latest_values_df, stale_after_n_days : int, id_column : str, reference_time = datetime.datetime.now(), reference_time_column = 'reference_time') -> set:
    # filter out entities with missing or stale values
    if latest_values_df.empty or stale_after_n_days is None:
        return set()
    
    latest_values_df = latest_values_df.copy()
    age = (reference_time - latest_values_df[reference_time_column]).dt.days
    stale = (age > stale_after_n_days)
    entities_with_invalid_attribute = set(latest_values_df[stale][id_column].to_numpy())
    return entities_with_invalid_attribute

def find_differring_rows(original_df, updated_df, key_column = 'None'):
    """
    Returns the index (or key column values) of rows that are different in the updated DataFrame compared to the original DataFrame.
    """
    if np.all(original_df.columns != updated_df.columns):
        raise ValueError(f"DataFrames must have the same columns to compute differences. Differing columns: {set(original_df.columns) ^ set(updated_df.columns)}")

    if key_column != 'None':
        if key_column not in original_df.columns:
            raise ValueError(f"Key column '{key_column}' must be present in both DataFrames.")

    # Create hashes per row
    original_df['__hash__'] = pd.util.hash_pandas_object(original_df, index=False)
    updated_df['__hash__'] = pd.util.hash_pandas_object(updated_df, index=False)

    hashes_df1 = set(original_df['__hash__'])
    hashes_df2 = set(updated_df['__hash__'])

    # Added or changed in updated_df
    changed_hashes = hashes_df2 - hashes_df1
    changed_rows = updated_df[updated_df['__hash__'].isin(changed_hashes)].drop(columns='__hash__')
    if key_column != 'None':
        return changed_rows[key_column].to_numpy()
    else:
        return changed_rows.index.to_numpy()


# TODO:
def parallelize_call(func : Callable, kwargs : dict, num_processes = 8):
    # find out what OS is used
    # if length of some key argument is less than threshold - just call the function with kwargs
    # otherwise split the kwargs into chunks and call the function in parallel
    chunked_kwargs = {}
    nonchunked_kwargs = {}
    for key in kwargs:
        if isinstance(kwargs[key], list) or isinstance(kwargs[key], set) or isinstance(kwargs[key], tuple) or isinstance(kwargs[key], np.ndarray):
            chunked_kwargs[key] = np.array_split(general_utils.to_array(kwargs[key]), num_processes)
        else:
            nonchunked_kwargs[key] = kwargs[key]
    
    partial_fn = partial(func, **nonchunked_kwargs)

    sig = inspect.signature(partial_fn)
    param_names = list(name for name in sig.parameters if name not in partial_fn.keywords)

    if len(param_names) == 1:
        map_args = chunked_kwargs[param_names[0]]
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.map(partial_fn, map_args)
        results = glue_outputs(results)
    elif len(param_names) > 1:
        split_args = [chunked_kwargs[name] for name in param_names]
        ready_args = [tuple(split_arg[c] for split_arg in split_args) for c in range(num_processes)]
        with multiprocessing.Pool(processes=num_processes) as pool:
            results = pool.starmap(partial_fn, ready_args)
        results = glue_outputs(results)
    else:
        results = partial_fn()

    return results

def glue_outputs(outputs):
    """
    Pomocna funkcia pre spojenie outputov rozdelenych pre paralelne spracovanie.

    Parameters
    ----------
    outputs : list of list | list of dict
        Zoznam outputov z paralelneho spracovania

    Returns
    -------
    list | dict
        Jednotlive outputy su spojene (concatenation) do jedneho listu alebo dict
    """
    # if the output is a list of lists, flatten it
    # if the output is a list of dicts, merge them
    if isinstance(outputs, list) and all(isinstance(i, list) | isinstance(i, np.ndarray) for i in outputs):
        if isinstance(outputs[0], list) or isinstance(outputs[0], str) or isinstance(outputs[0], np.ndarray):
            dtype = object
        else:
            dtype = type(outputs[0])
        return np.array([item for sublist in outputs for item in sublist], dtype=dtype)
    elif isinstance(outputs, list) and all(isinstance(i, tuple) for i in outputs):
        # concatenate each tuple separately
        result = []
        for tuple_idx in range(len(outputs[0])):
            list_for_tuple = [outputs[i][tuple_idx] for i in range(len(outputs))]
            result.append( glue_outputs(list_for_tuple) )
        return tuple(result)
    elif isinstance(outputs, list) and all(isinstance(i, dict) for i in outputs):
        result = {}
        for d in outputs:
            for k, v in d.items():
                if k not in result:
                    result[k] = []
                result[k].extend(v)
        # Convert lists to numpy arrays
        for k in result.keys():
            # if any of the elements are lists, use dtype=object; otherwise do not specify dtype
            if any(isinstance(x, list) for x in result[k]):
                result[k] = np.array(result[k], dtype=object)
            else:
                result[k] = np.array(result[k])
        return result
    elif isinstance(outputs, list) and all(isinstance(i, pd.DataFrame) for i in outputs):
        return pd.concat(outputs, ignore_index=True)
    else:
        raise ValueError("Outputs must be a list of lists/arrays/tuples/dicts.")

def remove_accents(text):
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if unicodedata.category(c) != 'Mn'  # Mn = Non-spacing mark (accents)
    )

def sanitize_timestamps(df: pd.DataFrame):
    datetime_columns = df.select_dtypes(include=['datetime64', '<M8']).columns
    sanitized_df = pd.DataFrame()
    for col in df.columns:
        if col in datetime_columns:
            sanitized_df.loc[:,col] = df.loc[:,col].dt.floor('s').astype('datetime64[ns]') 
        else:
            sanitized_df.loc[:,col] = df.loc[:,col]
    return sanitized_df

def try_parse(maybe_date : str):
    '''
    Try to parse a string into a datetime object. If it fails return None.
    '''
    if isinstance(maybe_date, datetime.datetime) or isinstance(maybe_date, pd.Timestamp):
        return maybe_date
    if not isinstance(maybe_date, str):
        return pd.NaT
    try:
        return dateutil.parser.parse(maybe_date, dayfirst = True)
    except:
        warnings.warn(f'try_parse failed to interpret: {maybe_date}')
        return pd.NaT
    

# TODO: introduce 'reference_time' string as argument?
# TODO: does not work for events type features, as they expect ids and this function does not allow for it
# TODO: move to FeatureStoreClient
'''
def DF_states_to_FS(df: pd.DataFrame, client: interface.FeatureStoreClient, path_to_feature_logic : str, reference_time_column: str, entity_column: str, calculation_time : datetime.datetime):
    """
    Take a DataFrame with feature values + column with entity_ids + column with reference_times and update each as individual features to the feature store.
    Only features with FeatureType == STATE are supported (as each feature is contained within one column).
    Features with FeatureType == EVENT have to be ingested using a different method (as each requires a specific id column definition).

    Names of columns are expected to be in the format:  feature_address = module_name.feature_name
    """

    if not entity_column in df.columns:
        raise ValueError(f"Entity column '{entity_column}' not found in input DataFrame.")
    
    if not reference_time_column in df.columns:
        raise ValueError(f"Reference time column '{reference_time_column}' not found in input DataFrame.")
    
    if not np.issubdtype(df[reference_time_column].dtype, np.datetime64):
        raise ValueError(f"Reference time column '{reference_time_column}' must be of datetime type. Got: {df[reference_time_column].dtype}")
    
    client.connect()

    value_columns = [col for col in df.columns if col not in [entity_column, reference_time_column]]
    
    for col in value_columns:
        feature = getFeatureMetaData(col, path_to_feature_logic)
        df_to_update = pd.DataFrame(
            {
                entity_column: df[entity_column],
                feature.name: df[col],
                reference_time_column: df[reference_time_column],
                'calculation_time': calculation_time
            }
        )

        df_reduced = df_to_update.sort_values([entity_column, reference_time_column])
        # keep the first record per entity and any record where value != previous value
        first_record_mask = df_reduced.groupby(entity_column).cumcount() == 0
        equal_values_mask = df_reduced[feature.name] == df_reduced.groupby(entity_column)[feature.name].shift()
        both_not_null_mask = df_reduced[feature.name].notnull() & df_reduced.groupby(entity_column)[feature.name].shift().notnull()
        changed_or_new_rows = (~equal_values_mask & both_not_null_mask) | first_record_mask
        df_reduced = df_reduced[changed_or_new_rows]

        client.write_feature(
            feature_name = feature.name,
            module_name = feature.module_name,
            feature_df = df_reduced)
'''