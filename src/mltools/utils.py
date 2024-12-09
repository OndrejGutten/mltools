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
