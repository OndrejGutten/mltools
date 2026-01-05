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

from param import output

from mltools.feature_store.core import Metadata
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

def check_fs_population(DB_credentials_yaml_path: str, start_date: datetime.datetime, end_date: datetime.datetime) -> pd.DataFrame:
    from sqlalchemy import inspect, text, create_engine
    import yaml
    with open(DB_credentials_yaml_path, 'r') as f:
        DB_creds = yaml.safe_load(f)
    engine = create_engine(f"postgresql+psycopg2://{DB_creds['feature_store_username']}:{DB_creds['feature_store_password']}@{DB_creds['feature_store_address']}")
    schema_name = 'features'
    sql_template = "SELECT DISTINCT reference_time FROM {table};"
    with engine.begin() as conn:
        inspector = inspect(conn)
        tables = inspector.get_table_names(schema=schema_name)
        results={}
        for table in tables:
            stmt = text(sql_template.format(table=f'"{schema_name}"."{table}"'))
            res = conn.execute(stmt)
            results[table] = res.fetchall()
            
    def list_datetimes(start: datetime, end: datetime):
        return [
            start + datetime.timedelta(days=i)
            for i in range((end - start).days + 1)
        ]
    output = pd.DataFrame(columns = list(results.keys()), index = list_datetimes(start_date, end_date))
    for k,v in results.items():
        for d in v:
            if d[0] not in output.index:
                continue
            output.loc[d[0],k] = True
    return output