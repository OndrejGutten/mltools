import pandas as pd
from sklearn.model_selection import train_test_split
import os
from . import utils
import mlflow


def load_data(config: dict):
    """
    Load data based on info from the config dictionary and return it as mlflow's dataset object.

    Parameters
    ----------
    config : dict
        A dictionary containing the configuration parameters. config['data'] is releveant for this function.

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

    ...     'data': {
    ...         'name': 'dataset_name',
    ...         'path': 'path/to/data', # relative to root if root is provided; absolute otherwise
    ...         'format': 'one_folder_per_sample',
    ...         'data_file': 'data.txt',
    ...         'label_file': 'label.txt', # optional
    ...     }
    ... }
    >>> dataset = load_data(config)
    """
    path_prefix = utils.get_nested(
        config, ['global', 'root_path'], '')
    data_path = utils.get_nested(config, ['data', 'path'])

    if data_path is None:
        raise ValueError('The data path is not provided.')

    full_data_path = os.path.join(path_prefix, data_path)

    dataset_name = utils.get_nested(
        config, ['data', 'name'], 'dataset')

    if config['data']['format'] == 'one_folder_per_sample':
        if not os.path.exists(full_data_path):
            raise FileNotFoundError(
                f'The folder {full_data_path} does not exist.')

        # Extract info on how to read the data
        # The data file is mandatory
        data_file = utils.get_nested(config, ['data', 'data_file'])
        if data_file is None:
            raise ValueError(
                'data_file not provided. For \'one_folder_per_sample\' format, a filename with data has to be provided in data/data_file entry.')

        # The label file is optional
        label_file = utils.get_nested(
            config, ['data', 'label_file'])

        # Read the data
        file_dict = {}
        file_dict['data'] = data_file
        if label_file is not None:
            file_dict['label'] = label_file
        df = datareader_one_folder_per_sample_to_df(full_data_path, file_dict)

        # Create the dataset object
        dataset_targets = 'label' if 'label' in file_dict else None
        dataset = mlflow.data.from_pandas(
            df, source=full_data_path, name=dataset_name, targets=dataset_targets)
        return dataset
    else:
        raise ValueError(f'Unsupported data format: \
                         {config["data"]["format"]}')


def datareader_one_folder_per_sample_to_df(src_folder: str, file_dict: dict):
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
    >>> df = one_folder_per_sample_to_df(src_folder, file_dict)
    """
    data = []
    for folder in os.listdir(src_folder):
        sample = {}
        for key, value in file_dict.items():
            with open(os.path.join(src_folder, folder, value), 'r') as f:
                sample[key] = f.read().strip()
        data.append(sample)

    return pd.DataFrame(data)


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
    -------
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
            mlflow.data.from_pandas(subset_dataset, name=f'{dataset.name} - {name}'))
    return tuple(mlflow_datasets)
