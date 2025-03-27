import os
from functools import reduce
import mlflow
import subprocess
import fsspec
import pandas as pd
import yaml

def parse_string_as_path(string: str) -> str | None:
    """
    Convert string to absolute path if it is non-empty. Check if the resulting path points to an existing file. If not, return None.

    Parameters
    ----------
    string : str
        Path to a file.

    Returns
    -------
    str or None
        Absolute path to a file or None.

    Examples
    --------
        >>> parse_string_as_path('config.yaml')
        '{current_path}/config.yaml'
        >>> parse_string_as_path('nonexistent_config.yaml')
        'None'
        >>> parse_string_as_path('')
        'None'
    """

    if string != '' and string is not None:
        tentative_filepath = os.path.join(os.getcwd(), string)
        if os.path.exists(tentative_filepath):
            return tentative_filepath
    return None


def get_nested(dictionary: dict, keys: list[str], default=None, verbose=True):
    '''
    Get a value from a nested dictionary by a list of keys. If the key is not found, return the default value.

    Parameters
    ----------
    dictionary : dict
        A nested dictionary.
    keys : list[str]
        A list of keys to traverse the dictionary.
    default : any, optional
        The default value to return if the key is not found. The default is None.
    verbose : bool, optional
        Print a message if the key is not found. The default is True.

    Returns
    -------
    any
        The value from the nested dictionary.

    Examples
    --------
        >>> get_nested({'a': {'b': {'c': 1}}}, ['a', 'b', 'c'])
        1
        >>> get_nested({'a': {'b': {'c': 1}}}, ['a', 'b', 'd'])
        None
    '''
    def namestr(obj, namespace):
        return [name for name in namespace if namespace[name] is obj]

    def reducer(d, key):
        try:
            if isinstance(d, dict):
                return d[key]
            raise KeyError
        except KeyError:
            if verbose:
                print(f"Key '{keys}' not found in {namestr(d, globals())}")
            return default

    value = reduce(reducer, keys, dictionary)
    return value


def set_run_tags(config: dict):
    """
    Set tags for the current MLflow run.

    Parameters
    ----------
    config : dict
        A dictionary with at least the following keys:
        - 'author': str
        - 'description': str
        - 'task_type': str
        - 'project_name': str

    Returns
    -------
    None

    Examples
    --------
        >>> config = {
        ...     'author': Boaty McBoatface',
        ...     'description': 'KNN classifier for the iris dataset',
        ...     'task_type': 'classification',
        ...     'project_name': 'training'
        ...     'custom_tag': 'custom_value'   
        ... }
        >>> set_run_tags(config)
    """
    author = get_nested(config, ['experiment_metadata', 'author'], None)
    description = get_nested(config, ['experiment_metadata', 'description'], None)
    task_type = get_nested(config, ['experiment_metadata', 'task_type'], None)
    project_name = get_nested(config, ['experiment_metadata', 'project_name'], None)

    if author is None:
        raise ValueError("Author not provided.")
    if description is None:
        raise ValueError("Description not provided.")
    if task_type is None:
        raise ValueError("Task type not provided.")
    if project_name is None:
        raise ValueError("Project name not provided.")

    for key, value in config['experiment_metadata'].items():
        if key == 'description':
            key = 'mlflow.note.content'
        # if value is a dictionary raise error
        if isinstance(value, dict):
            raise ValueError(f"Value for key '{key}' cannot be a dictionary.")
        mlflow.set_tag(key, str(value))


def generate_requirements_file(file = 'requirements.txt'):
    """
    Generate a requirements.txt file based on the current environment's dependencies.

    Parameters
    ----------
    file : str, optional
        The name of the requirements file. The default is 'requirements.txt'.
    Returns
    -------
    None
    """

    # Run `pip freeze` to capture the current environment's dependencies
    result = subprocess.run(["pip", "freeze"], stdout=subprocess.PIPE, text=True)
    
    # Write the output to `requirements.txt`
    with open("requirements.txt", "w") as req_file:
        req_file.write(result.stdout)

def path_exists(uri: str) -> bool:
    """
    Check if a file/directory exists. Supports multiple filesystems (e.g. local, http, s3, gs, etc.)

    Parameters
    ----------
    uri : str
        A path to check.

    Returns
    -------
    bool
        True if the path exists, False otherwise.

    Examples
    --------
        >>> path_exists('data')
        True
        >>> path_exists('nonexistent_data')
        False
        >>> path_exists('file:///data')
        True
        >>> path_exists('http://127.0.0.1/data')
        True
    """
    pass
    try:
        protocol = fsspec.utils.get_protocol(uri)
        fs = fsspec.filesystem(protocol)
    except ValueError:
        raise ValueError(f"Protocol not recognized: {protocol}")
        
    return fs.exists(uri)

def make_iterable(obj : any, type: type = list) -> list | tuple | pd.DataFrame:
    """
    Convert an object to an iterable.

    Parameters
    ----------
    obj : any
        An object to convert to an iterable.
    type : type, optional
        The type of the iterable. The default is list.

    Returns
    -------
    iterable
        An iterable object.

    Examples
    --------
        >>> make_iterable('a')
        ['a']
        >>> make_iterable(['a'])
        ['a']
        >>> make_iterable('a', pd.DataFrame)
        pd.DataFrame(['a'])
        >>> make_iterable(['a'], pd.DataFrame)
        pd.DataFrame(['a'])
        >>> make_iterable(pd.DataFrame(['a']))
        ['a']
    """
    if isinstance(obj, type):
        return obj

    if isinstance(obj,pd.DataFrame):
        obj = obj.values.flatten().tolist()

    if type == list and (not hasattr(obj, '__iter__') or isinstance(obj, str)):
        return [obj]

    if type == pd.DataFrame and (not hasattr(obj, '__iter__') or isinstance(obj, str)):
        return pd.DataFrame([obj])
    return type(obj)

def get_dependencies_from_MLProject(project_file: str = 'MLproject') -> dict:
    """
    Extract dependencies from an MLproject file.

    Parameters
    ----------
    file : str, optional
        The name of the MLproject file. The default is 'MLproject'.

    Returns
    -------
    list[str]
        A list of dependencies as listed in the environment file referenced in the MLproject file.

    Examples
    --------
        >>> get_dependencies_from_MLProject('MLproject')
        ['pandas', 'scikit-learn', 'numpy']
    """
    # Path to the MLproject file
    mlproject_path = os.path.join(project_file)

    # Read the MLproject file
    with open(mlproject_path, "r") as f:
        mlproject = yaml.safe_load(f)

    # Extract the environment file name
    env_file = mlproject.get("conda_env", mlproject.get(
        "pip_env", mlproject.get("python_env", None)))
    if not env_file:
        raise ValueError("No environment file specified in MLproject")

    # Load the environment file content
    with open(env_file, "r") as f:
        env_content = yaml.safe_load(f)
    # Extract pip dependencies (if conda_env)
    return env_content.get("dependencies", [])



def get_run_name_from_model_uri(model_uri: str) -> str:
    """
    Extract the run name from a model URI.

    Parameters
    ----------
    model_uri : str
        The URI of the model.

    Returns
    -------
    str
        The run name.

    Examples
    --------
        >>> get_run_name_from_model_uri('runs:/abc123def/model')
        'run_name'
    """
    model_info = mlflow.models.get_model_info(model_uri)
    run_id = model_info.run_id
    run_info = mlflow.get_run(run_id)
    return run_info.data.tags.get("mlflow.runName", None)

def mlflow_get_logged_model_uris(run_id):
    """
    Get all paths to models logged during a run. Only the 

    Parameters
    ----------
    run_id : str
        The ID of the run.

    Returns
    -------
    list
        A list of models logged during the run.

    Examples
    --------
        >>> mlflow_get_logged_models('abc123def')
        ['model1', 'model2']
    """
    client = mlflow.tracking.MlflowClient()
    artifacts = client.list_artifacts(run_id)

    # Check for directories that indicate logged models
    logged_model_uris = []
    for artifact in artifacts:
        if artifact.is_dir and client.list_artifacts(run_id, artifact.path):
            # Check if the directory contains an MLmodel file
            sub_artifacts = client.list_artifacts(
                run_id, artifact.path)
            if any(a.path.endswith("MLmodel") for a in sub_artifacts):
                model_info = mlflow.models.get_model_info(f'runs:/{run_id}/{artifact.path}')
                logged_model_uris.append(model_info.model_uri)

    return logged_model_uris

def read_class_weights_from_file(filepath: str) -> dict:
    """
    Read class weights from a file.

    Parameters
    ----------
    file : str
        The path to the file.

    Returns
    -------
    dict
        A dictionary with class weights.

    Examples
    --------
        >>> read_class_weights_from_file('class_weights.yaml')
        {'class1': 0.1, 'class2': 0.2}
    """
    if not os.path.exists(filepath):
        raise ValueError(f"Class weights file not found at {filepath}")
    
    class_weights = pd.read_csv(filepath, header=None)
    if class_weights.shape[1] != 2:
        raise ValueError("Class weights file must have 2 columns - class label (first) and its weight (second).")
    class_weights_dict = {str(k):v for k,v in zip(class_weights.iloc[:,0], class_weights.iloc[:,1])}
    return class_weights_dict

def read_cost_matrix_from_file(filepath: str) -> pd.DataFrame:
    """
    Read a cost matrix from a file.

    Parameters
    ----------
    file : str
        The path to the file. First line and the first column must contain matching labels.

    Returns
    -------
    pd.DataFrame
        A cost matrix.

    Examples
    --------
        % cat cost_matrix.csv
        ,class1,class2
        class1,0,1
        class2,1,0
        >>> read_cost_matrix_from_file('cost_matrix.csv')
        pd.DataFrame([[0, 1], [1, 0]])

        % cat cost_matrix.csv
        ,class1,class2,class3
        class1,0,1,1
        class3,1,0,1
        class2,1,1,0

        >>> read_cost_matrix_from_file('cost_matrix.csv')
        ValueError: Cost matrix must have the same index and columns.

    """
    if not os.path.exists(filepath):
        raise ValueError(f"Cost matrix file not found at {filepath}")
    
    cost_matrix = pd.read_csv(filepath, header = 0, index_col = 0)
    cost_matrix.columns = cost_matrix.columns.astype(str)
    cost_matrix.index = cost_matrix.index.astype(str)
    validate_cost_matrix(cost_matrix)
    return cost_matrix

def validate_cost_matrix(cost_matrix: pd.DataFrame):
    if cost_matrix.shape[0] != cost_matrix.shape[1]:
        raise ValueError("Cost matrix must be a square.")
    
    if not all(cost_matrix.index.astype(str) == cost_matrix.columns.astype(str)):
        raise ValueError("Cost matrix must have the same index and columns.")
    
def get_unique_experiment_id_from_name(experiment_name: str):
    experiments = mlflow.search_experiments(filter_string = f'name = "{experiment_name}"')
    if len(experiments) == 0:
        raise Exception(f'No experiment named {experiment_name} found')
    elif len(experiments) > 1:
        raise Exception(f'Multiple experiments named {experiment_name} found')
    else:
        return experiments[0].experiment_id

def attach_metadata_to_pandas_dataframe(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """
    Attach metadata to a pandas DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame.
    metadata : dict
        A dictionary with metadata.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with metadata attached.

    Examples
    --------
        >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> metadata = {'author': 'Boaty McBoatface', 'description': 'A dataset with columns a and b'}
        >>> attach_metadata_to_pandas_dataframe(df, metadata).attrs
        {'author': 'Boaty McBoatface', 'description': 'A dataset with columns a and b'}
    """
    for key, value in metadata.items():
        df.attrs[key] = value
    return df