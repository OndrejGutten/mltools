import os
from functools import reduce
import mlflow
import subprocess


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
        A dictionary with the following keys:
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
        ...     'author': '
        ...     'description': 'A simple experiment',
        ...     'task_type': 'classification',
        ...     'project_name': 'experiment_logging'
        ... }
        >>> set_run_tags(config)
    """
    mlflow.set_tag('author', get_nested(
        config, ['experiment_metadata', 'author']))
    mlflow.set_tag('project_name', get_nested(
        config, ['experiment_metadata', 'project_name']))
    mlflow.set_tag('mlflow.note.content', get_nested(
        config, ['experiment_metadata', 'description']))
    mlflow.set_tag('task_type', get_nested(
        config, ['experiment_metadata', 'task_type']))


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
