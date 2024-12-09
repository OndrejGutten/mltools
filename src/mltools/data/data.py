import pandas as pd
from sklearn.model_selection import train_test_split
import os
from mltools import utils
import mlflow


def read_data(config: dict):
    #TODO: always return a list of datasets (even if split_sizes argument is not provided)
    #TODO how to deal with multiple datasets when sometimes we need input/output but only read input
    """
    Load data based on info from the config dictionary and return it as mlflow's dataset object.

    Parameters
    ----------
    config : dict
        A dictionary containing the configuration parameters. config['dataset'] is releveant for this function.

    Returns
    -------
    mlflow.data.PandasDataset
        The loaded data.

    Examples
    --------
    >>> config = {
            'global': {
                root_path: 'path/to/root'
            }

    ...     'dataset': {
    ...         'name': 'dataset_name',
    ...         'path': 'path/to/data', # relative to root if root is provided; absolute otherwise
    ...         'format': 'one_folder_per_sample',
    ...         'target': 'label' # Name of the column where labels are stored. Optional. 
    ...         'mapping': { // relevant only for 'one_folder_per_sample' format, see corresponding datareader/datawriter function
    ...             'data': 'data.txt',
    ...             'label': 'data.txt'
    ...         }
    ...     }
    ... }
    >>> dataset = load_data(config)
    """

    dataset_block = utils.get_nested(config, ['dataset'], None)
    datasets_block = utils.get_nested(config, ['datasets'], None)

    if dataset_block is None and datasets_block is None:
        raise ValueError('The dataset or datasets block is not provided - config[dataset] or config[datasets]')
    
    if dataset_block is not None and datasets_block is not None:
        raise ValueError('Both dataset and datasets blocks are provided - only one of config[dataset]/config[datasets] is allowed')
    
    global_subblock = utils.get_nested(config, ['global'], {})

    if dataset_block is not None:
        return _read_single_dataset_from_config_subblock(dataset_block, global_subblock)
    else:
        return [_read_single_dataset_from_config_subblock(subblock, global_subblock) for subblock in datasets_block]


def _read_single_dataset_from_config_subblock(single_dataset_subblock: dict, global_subblock: dict):
    path_prefix = utils.get_nested(
        global_subblock, ['root_path'], '')
    data_path = utils.get_nested(single_dataset_subblock, ['path'], None)

    if data_path is None:
        raise ValueError('The data path is not provided.')

    full_data_path = os.path.join(path_prefix, data_path)

    dataset_name = utils.get_nested(
        single_dataset_subblock, ['name'], 'dataset')

    format_subblock = utils.get_nested(single_dataset_subblock, ['format'], None)

    if format_subblock == 'one_folder_per_sample':
        # Read the data
        file_to_value_mapping = utils.get_nested(single_dataset_subblock, ['mapping'], {})
        df = datareader_one_folder_per_sample(full_data_path, file_to_value_mapping)
    elif format_subblock == 'mlflow_run':
        raise NotImplementedError('The format \'mlflow_run\' is not implemented yet.')
    elif format_subblock == 'from_file':
        if not os.path.exists(full_data_path):
            raise FileNotFoundError(
                f'The file {full_data_path} does not exist.')
        
        df = datareader_from_file(full_data_path)
    else:
        raise ValueError(f'Unsupported data format: \
                         {format_subblock}')

    # Create the dataset object
    dataset_target = utils.get_nested(single_dataset_subblock, ['targets'], None)
    dataset = mlflow.data.from_pandas(
        df, source=full_data_path, name=dataset_name, targets=dataset_target)
    
    # split the data if split_sizes are provided
    split_sizes = utils.get_nested(single_dataset_subblock, ['split_sizes'], None)
    if split_sizes is not None:
        datasets = mlflow_split_categorical_data(dataset, sizes=split_sizes, stratified=True, target_column=dataset_target)
        return datasets

    return dataset

def write_data(config: dict, dataset:  mlflow.data.pandas_dataset.PandasDataset):
    # TODO: if index is not reset it will crash
    # TODO: if mapping is incomplete, unmentioned columns will be written verbatim - is this desired?
    '''
    Write data based on info from the config dictionary.

    Parameters
    ----------
    config : dict
        A dictionary containing the configuration parameters. config[output] is releveant for this function.

    Returns
    -------
    None

    Examples
    --------
    >>> config = {
    ...     'global': {
    ...         'root_path': 'path/to/root'
    ...     },
    ...     'output': {
    ...         'path': 'path/to/output',
    ...         'format': 'one_folder_per_sample', // or 'to_file'
    ...         'file_dict': { // relevant only for 'one_folder_per_sample' format, see corresponding datawriter function
    ...             'text': 'Text.txt',
    ...             'label': 'Type.txt'
    ...         }
    ...     }
    ... }
    >>> dataset = mlflow.data.PandasDataset(pd.DataFrame({
    ...     'text': ['a', 'b', 'c', 'd', 'e'],
    ...     'label': [0, 1, 0, 1, 0]
    ... }))
    >>> write_data(config, dataset) 
    '''

    path_prefix = utils.get_nested(
        config, ['global', 'root_path'], '')
    output_path = utils.get_nested(config, ['output', 'path'])
    output_format = utils.get_nested(config, ['output', 'format'], None)
    output_file_mapping = utils.get_nested(config, ['output', 'mapping'], {})

    if output_path is None:
        raise ValueError('The output path is not provided - config[output][output_path]')

    if output_format is None:
        raise ValueError('The output format is not provided - config[output][format]')
    
    if len(output_file_mapping) == 0 and output_format == 'one_folder_per_sample':
        raise ValueError('The file mapping is not provided - config[output][mapping]')

    full_output_path = os.path.join(path_prefix, output_path)

    if output_format == 'one_folder_per_sample':
        datawriter_one_folder_per_sample(dataset.df, full_output_path, output_file_mapping)
    elif output_format == 'to_file':
        datawriter_to_file(dataset.df, full_output_path)
    else:
        raise ValueError(f'Unsupported output format: \
                         {output_format}')

def datawriter_one_folder_per_sample(df: pd.DataFrame, path: str, file_dict: dict = {}):
    '''
    Write data from a DataFrame to a folder. If the folder contains any files an error is raised.
    Each sample is written into a separate subfolder. Each subfolder files named after DataFrame column names or mapped by file_dict parameter.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to write.
    path : str
        The path to the root folder where the data will be written.
    file_dict : dict
        An optional map of column names to file names. If not provided, the column names are used as file names with *.txt extension.

    Returns
    -------
    None

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'text': ['a', 'b', 'c', 'd', 'e'],
    ...     'label': [0, 1, 0, 1, 0]
    ... })
    >>> path = 'empty_folder'
    >>> datawriter_one_folder_per_sample(df, path)
        'Data written'
    >>> path = 'folder_with_files'
    >>> datawriter_one_folder_per_sample(df, path)
        'Error: Folder is not empty'
    >>> path = 'non_existing_folder'
    >>> datawriter_one_folder_per_sample(df, path)
        'Data written'
    >>> path = 'empty_folder_with_custom_files'
    >>> file_dict = {
    ...     'text': 'Text.txt'
    ... }
    >>> datawriter_one_folder_per_sample(df, path, file_dict)
        'Data written'
    '''
    if os.path.exists(path) and len(os.listdir(path)) > 0:
        raise ValueError('Error: Folder is not empty')

    for i, row in df.iterrows():
        sample_folder = os.path.join(path, str(df.index[i]))
        os.makedirs(sample_folder, exist_ok=True)
        for column, value in row.items():
            mapped_name = file_dict.get(column, f'{column}.txt')
            with open(os.path.join(sample_folder, mapped_name), 'w') as f:
                f.write(str(value))

def datawriter_to_file(df: pd.DataFrame, file_path: str):
    '''
    Write data from a DataFrame to a file. Format is inferred from the file extension.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to write.
    file_path : str
        The path to the file where the data will be written. The extension of the file determines the format. Supported formats are: .csv, .pkl, .pickle, .joblib.

    Returns
    -------
    None

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'text': ['a', 'b', 'c', 'd', 'e'],
    ...     'label': [0, 1, 0, 1, 0]
    ... })
    >>> file_path = 'data.csv'
    >>> datawriter_from_df_to_file(df, file_path)
        'Data written'
    >>> file_path = 'data.xlsx'
    >>> datawriter_from_df_to_file(df, file_path)
        'Error: Unsupported file format'
    '''

    extension = os.path.splitext(file_path)[1]

    if extension not in ['.csv', '.pkl', '.pickle', '.joblib']:
        raise NotImplementedError(f'Unsupported file format: {extension}')

    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    if extension == '.csv':
        df.to_csv(file_path, index=False)
    elif extension == '.pkl' or extension == '.pickle':
        df.to_pickle(file_path)
    elif extension == '.joblib':
        joblib.dump(df, file_path)
    else:
        raise NotImplementedError(f'Unsupported file format: {extension}')

def datareader_one_folder_per_sample(src_folder: str, file_dict: dict):
    """
    Parameters
    ----------
    src_folder : str
        The path to the root folder containing the samples.
    file_dict : dict
        A dictionary containing the file names storing the values of parameters for each sample.
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the data from the specified folder.
    Examples
    --------
    >>> src_folder = 'data'
    >>> file_dict = {
    ...     'text': 'Text.txt',
    ...     'label': 'Type.txt'
    ... }
    >>> df = one_folder_per_sample(src_folder, file_dict)
    """
    data = []
    for folder in os.listdir(src_folder):
        sample = {}
        for key, value in file_dict.items():
            with open(os.path.join(src_folder, folder, value), 'r') as f:
                sample[key] = f.read().strip()
        data.append(sample)

    return pd.DataFrame(data)

def datareader_from_file(file_path: str):
    """
    Parameters
    ----------
    file_path : str
        The path to the file containing the data.
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the data from the specified file.
    Examples
    --------
    >>> file_path = 'data.csv'
    >>> df = datareader_from_file(file_path)
    """
    extension = os.path.splitext(file_path)[1]
    if extension == '.csv':
        df = pd.read_csv(file_path, header = 0)
    elif extension == '.parquet':
        df = pd.read_parquet(file_path)
    elif extension == '.pkl' or extension == '.pickle':
        df = pd.read_pickle(file_path)
    elif extension == '.joblib':
        df = joblib.load(file_path)
    else:
        raise NotImplementedError(f'Unsupported file format: {extension}')
    return df

def datareader_from_mlflow_run(run_id):
    # TODO:
    pass



def pandas_split_categorical_data(df: pd.DataFrame, sizes: list[float] = [0.8, 0.2], stratified: bool = False, target_column: str = None, random_state: int = 42):
    """
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to split.
    sizes : list of float
        The proportions of the data to include in the individual splits. The sum of the sizes must be 1.
    stratified : bool
        Whether to perform stratified sampling.
    target_column : str
        The name of the column containing the target values.
    random_state : int
        The seed used by the random number generator.
    Returns
    -------d
    tuple of pd.DataFrame
        The train, validation, and test sets.
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'text': ['a', 'b', 'c', 'd', 'e'],
    ...     'label': [0, 1, 0, 1, 0]
    ... })
    >>> train, val, test = split_categorical_data(df, sizes=[0.6, 0.2, 0.2], stratified=True, target_column='label')
    """

    if sum(sizes) != 1:
        raise ValueError('The sum of the sizes must be 1.')

    subset_dfs = []

    remaining_df = df
    for _, size in enumerate(sizes[:-1]):
        if stratified:
            subset_df, remaining_df = train_test_split(
                remaining_df, test_size=1 - size, stratify=remaining_df[target_column], random_state=random_state)
        else:
            subset_df, remaining_df = train_test_split(
                remaining_df, test_size=1 - size, random_state=random_state)
        subset_dfs.append(subset_df)

    subset_dfs.append(remaining_df)

    return tuple(subset_dfs)


def mlflow_split_categorical_data(dataset: mlflow.data.pandas_dataset.PandasDataset, sizes: list[float] = [0.8, 0.2], stratified: bool = False, target_column: str = None, random_state: int = 42):
    """
    Parameters
    ----------
    dataset : mlflow.data.PandasDataset
        The dataset to split.
    sizes : list of float
        The proportions of the data to include in the individual splits. The sum of the sizes must be 1.
    stratified : bool
        Whether to perform stratified sampling.
    target_column : str
        The name of the column containing the target values.
    random_state : int
        The seed used by the random number generator.
    Returns
    -------
    tuple of mlflow.data.PandasDataset
        The train, validation, test, test2, test3, etc. sets.
    Examples
    --------
    >>> dataset = mlflow.data.PandasDataset(pd.DataFrame({
    ...     'text': ['a', 'b', 'c', 'd', 'e'],
    ...     'label': [0, 1, 0, 1, 0]
    ... }))
    >>> train, val, test = mlflow_split_categorical_data(dataset, sizes=[0.6, 0.2, 0.2], stratified=True, target_column='label')
    """
    subset_dfs = pandas_split_categorical_data(
        dataset.df, sizes=sizes, stratified=stratified, target_column=target_column, random_state=random_state)
    names = ['train', 'validation', 'test']
    for i in range(3, len(sizes)):
        names.append('test' + str(i - 1))
    mlflow_datasets = []
    for subset_dataset, name in zip(subset_dfs, names):
        mlflow_datasets.append(
            mlflow.data.from_pandas(subset_dataset, name=f'{dataset.name} - {name}', source=mlflow.data.get_source(dataset), targets=dataset.targets))
    return tuple(mlflow_datasets)

def drop_classes_with_1_member(df: pd.DataFrame, target_column: str):
    """
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to process.
    target_column : str
        The name of the column containing the target values.
    Returns
    -------
    pd.DataFrame
        The DataFrame with classes that have only one member removed.
    Examples
    --------
    >>> df = pd.DataFrame({
    ...     'text': ['a', 'b', 'c', 'd', 'e'],
    ...     'label': [0, 1, 0, 1, 0]
    ... })
    >>> df = drop_classes_with_1_member(df, 'label')
    """
    class_counts = df[target_column].value_counts()
    classes_to_drop = class_counts[class_counts == 1].index
    return df[~df[target_column].isin(classes_to_drop)]