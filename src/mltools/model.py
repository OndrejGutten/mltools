import mltools
import mlflow
import pandas as pd
import os 
import yaml

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer

import mltools.logging

def load_model(config = None, model_uri = None, model_id = None):
    """
    Load a model using a config or model_uri or model_id. Only one may be provided.

    Parameters
    ----------
    config : dict
        A dictionary containing the configuration parameters. config['model'] is relevant for this function. It must contain the uri of the model, any other entries will be ignored. If provided, model_uri and model_id must not be provided.
    model_uri : str
        The URI of the model. If provided, config must not be provided.
    model_id : str
        The ID of the model. Stored as a 'model_id' tag in mlflow registry. If provided, config and model_uri must not be provided.

    Returns
    -------
    sklearn model
        The model
    Examples
    --------
    >>> config = {
    ...     'model': {
    ...         'uri': model_uri
    ...     }
    ... }
    >>> model, model_info = load_model(config)
    """

    if (config is not None) + (model_uri is not None) + (model_id is not None) != 1:
        raise ValueError("Exactly one of config, model_uri or model_id must be provided.")
    
    if model_id is not None:
        # find model by tag
        all_models = mlflow.search_model_versions()
        matching_models = [model for model in all_models if 'model_id' in model.tags and model.tags['model_id'] == model_id]
        if len(matching_models) == 0:
            raise ValueError(f"No model found with model_id {model_id}.")
        elif len(matching_models) > 1:
            raise ValueError(f"Multiple models found with model_id {model_id}. Please specify a more specific identifier.")
        else:
            model_uri = matching_models[0].source + '/model'
    elif model_uri is None:
        pyfunc_model = mlflow.pyfunc.load_model(model_uri)
        return pyfunc_model._PyFuncModel__model_impl.python_model.model # unwrap the model from the pyfunc wrapper
    elif config is not None:
        model_uri = mltools.utils.get_nested(config, ['model', 'uri'], None)
        if model_uri is None:
            raise ValueError("Model uri not provided.")

        pyfunc_model = mlflow.pyfunc.load_model(model_uri)
        return pyfunc_model._PyFuncModel__model_impl.python_model.model # unwrap the model from the pyfunc wrapper


def setup_model(config):
    """
    Setup a model given information in the config. If uri is provided, load the model from the uri. Otherwise, create a new model.

    Parameters
    ----------
    config : dict
        A dictionary containing the configuration parameters. config['model'] is relevant for this function.

    Returns
    -------
    mlflow.pyfunc.PyFuncModel
        The model.
    """
    path_prefix = mltools.utils.get_nested(config, ['global','root_path'], '')

    model_uri = mltools.utils.get_nested(config, ['model', 'uri'], None)
    if model_uri is not None:
        return load_model(model_uri)

    model_type = mltools.utils.get_nested(config, ['model', 'type'], None)
    if model_type is None:
        raise ValueError("Model type not provided.")

    model_name = mltools.utils.get_nested(config, ['model', 'name'], None)
    parameters = mltools.utils.get_nested(config,['model','parameters'], {})

    ### class weights options
    has_class_weights = False
    class_weights_config_dict = mltools.utils.get_nested(config, ['model','class_weights'], None)
    class_weights_dict = None

    if class_weights_config_dict is not None:
        has_class_weights = True
        class_weights_relative_or_absolute = mltools.utils.get_nested(class_weights_config_dict, ['relative_or_absolute'], 'absolute')

        class_weights_file_path = mltools.utils.get_nested(class_weights_config_dict, ['file_path'], None)
        if class_weights_file_path is None:
            raise ValueError("Class weights file path not provided.")
        
        class_weights_full_file_path = os.path.join(path_prefix, class_weights_file_path)
        class_weights_dict = mltools.utils.read_class_weights_from_file(class_weights_full_file_path)

    if model_type == "TF-IDF-KNN":
        baseline_classifier = KNeighborsClassifier(**parameters)
        model = mltools.architecture.TF_IDF_Classifier(baseline_classifier = baseline_classifier, model_name = model_name)
    elif model_type == "TF-IDF-SVM":
        baseline_classifier = SVC(probability = True, **parameters)
        model = mltools.architecture.TF_IDF_Classifier(baseline_classifier = baseline_classifier, model_name = model_name)
        if has_class_weights:
            model.set_class_weights(class_weights_dict)
    elif model_type == "TF-IDF-RF":
        baseline_classifier = RandomForestClassifier(**parameters)
        model = mltools.architecture.TF_IDF_Classifier(baseline_classifier = baseline_classifier, model_name = model_name)
        if has_class_weights:
            model.set_class_weights(class_weights_dict)
    elif model_type == "TF-IDF-XGBoost":
        # load cost_matrix
        cost_matrix = None
        cost_matrix_filepath = mltools.utils.get_nested(config,['model','cost_matrix'], None)
        if cost_matrix_filepath is not None:
            cost_matrix_filepath = os.path.join(path_prefix, cost_matrix_filepath)
            cost_matrix = mltools.utils.read_cost_matrix_from_file(cost_matrix_filepath)

        model = mltools.architecture.TF_IDF_XGBoost_Classifier(model_name = model_name, parameters = parameters, class_weights = class_weights_dict, cost_matrix = cost_matrix)
    elif model_type == "TextPreprocessor":
        preprocessor = mltools.architecture.TextPreprocessor(**parameters)
        # prepare a version with required signature for mlflow
        transformer = FunctionTransformer(
            preprocessor.preprocess_iterable, kw_args={'preprocess_stack': config['model']['parameters']['preprocess_stack']})
        model = mltools.architecture.PredictTransformerWrapper(transformer, transforms_X = True, transforms_y = False)
        return model
    elif model_type == 'Relabeller':
        preprocessor = mltools.architecture.Relabeller(**parameters)
        transformer =  FunctionTransformer(preprocessor.predict)
        model = mltools.architecture.PredictTransformerWrapper(preprocessor, transforms_X = False, transforms_y = True)
        return model
    else:
        raise ValueError(f"Model type {model_type} not recognized.")
    
    return model

def load_model_list(config):
    """
    Load models from the uri provided in the config.

    Parameters
    ----------
    config : dict
        A dictionary containing the configuration parameters. config['model_list'] is relevant for this function.

    Returns
    -------
    list
        A list of models.
    list
        A list of mlflow Model objects
    
    Examples
    --------
    >>> config = {
    ...     'model_list': [ model_1_uri, model_2_uri, model_3_uri ]
    ... }
    >>> models, models_metadata = load_model_list(config)
    """
    return [load_model({'model': {'uri' : model_uri}}) for model_uri in config['model_list']]

def load_model_info(config):
    """
    Load model info from the uri provided in the config.

    Parameters
    ----------
    config : dict
        A dictionary containing the configuration parameters. config['model'] is relevant for this function.

    Returns
    -------
    mlflow.models.Model
        Metadata about the model
    
    Examples
    --------
    >>> config = {
    ...     'model': {
    ...         'uri': model_uri
    ...     }
    ... }
    >>> model_info = load_model_info(config)
    """
    model_uri = mltools.utils.get_nested(config, ['model', 'uri'], None)
    if model_uri is None:
        raise ValueError("Model uri not provided.")

    return mlflow.models.get_model_info(model_uri)

def load_model_info_list(config):
    """
    Load model info from the uri provided in the config.

    Parameters
    ----------
    config : dict
        A dictionary containing the configuration parameters. config['model_list'] is relevant for this function.

    Returns
    -------
    list
        A list of metadata about the models
    
    Examples
    --------
    >>> config = {
    ...     'model_list': [ model_1_uri, model_2_uri, model_3_uri ]
    ... }
    >>> models_metadata = load_model_info_list(config)
    """
    return [load_model_info({'model': {'uri' : model_uri}}) for model_uri in config['model_list']]

def model_uri_to_version(model_uri):
    """
    Extract the version from a model URI.

    Parameters
    ----------
    model_uri : str
        The URI of the model.

    Returns
    -------
    str
        The version of the model.
    
    Examples
    --------
    >>> model_uri_to_version('runs:/abc123def/model')
    'abc123def'
    """
    model_info = mlflow.models.get_model_info(model_uri)
    return model_info.run_id

def download_model_artifact(artifact_name, model_uri = None, model_name = None, model_version = None, model_alias = None):
    """
    Download an artifact stored in the model's home path.

    Parameters
    ----------
    artifact_name : str
        The name of the artifact to load.
    model_uri : str
        The URI of the model. If provided, other model parameters must not be provided.
    model_name : str, optional
        The name of the model. If provided, either model_version or model_alias must also be provided.
    model_version : str, optional
        The version of the model. Only to be used with model_name. If provided, model alias must not be provided.
    model_alias : str, optional
        The alias of the model. Only to be used with model_name. If provided, model version must not be provided.
    
    Returns
    -------

    """

    if not mltools.logging.is_remote_mlflow_server_running():
        raise RuntimeError("MLflow server is not running. Cannot load model artifact.")

    if model_uri is not None:
        if model_name is not None or model_version is not None or model_alias is not None:
            raise ValueError("If model_uri is provided, model_name, model_version and model_alias must not be provided.")
        model_info = mlflow.models.get_model_info(model_uri)
        artifact = mlflow.artifacts.download_artifacts(run_id = model_info.run_id, artifact_path = f'{model_info.artifact_path}/{artifact_name}')
        return artifact
    else:
        if model_name is None:
            raise ValueError("Either model_uri or model_name+version/alias must be provided.")
        if model_version is None and model_alias is None:
            raise ValueError("If model_name is provided, either model_version or model_alias must be provided.")
        if model_version is not None and model_alias is not None:
            raise ValueError("If model_name is provided, either model_version or model_alias must be provided, not both.")
        if model_version is not None:
            client = mlflow.MlflowClient()
            model_version = client.get_model_version(model_name, model_version)
        elif model_alias is not None:
            client = mlflow.MlflowClient()
            model_version = client.get_model_version_by_alias(model_name, model_alias)

        # TODO: list artifacts first and check if the artifact exists

        artifact = mlflow.artifacts.download_artifacts(artifact_uri = model_version.source + '/' + artifact_name)
        return artifact

def get_model_name_from_uri(uri: str) -> str:
    """
    Extract the model name from a model URI.

    Parameters
    ----------
    uri : str
        The URI of the model.
    Returns
    -------
    str
        The name of the model.
    """

    if uri.startswith("models:/"):
        uri_body = uri[len("models:/"):]
        if "@" in uri_body:
            model_name = uri_body.split("@")[0]
        elif "/" in uri_body:
            model_name = uri_body.split("/")[0]
        else:
            model_name = uri_body
        return model_name
    else:
        raise ValueError(f"Unsupported model URI format: {uri}")
    
def model_uri_to_run_id(uri: str) -> str:
    """
    Extract the run ID from a model URI.

    Parameters
    ----------
    uri : str
        The URI of the model.

    Returns
    -------
    str
        The run ID of the model.

    Raises
    ------
    ValueError
        If the URI does not contain a run ID.
    """

    if not mltools.logging.is_remote_mlflow_server_running():
        raise RuntimeError("MLflow server is not running. Cannot extract run ID.")
    
    model_info = mlflow.models.get_model_info(uri)

    return model_info.run_id

def run_id_to_model_uri(run_id: str) -> str:
    """
    Convert a run ID to a model URI.

    Parameters
    ----------
    run_id : str
        The run ID of the model.

    Returns
    -------
    str
        The model URI.
    
    Raises
    ------
    ValueError
        If the run ID is not found.
    """
    
    if not mltools.logging.is_remote_mlflow_server_running():
        raise RuntimeError("MLflow server is not running. Cannot convert run ID to model URI.")

    run_id_associated_with_model = run_id in [version.run_id for version in mlflow.search_model_versions()]

    if run_id_associated_with_model is False:
        raise ValueError(f"Run ID {run_id} is not associated with any model.")
    
    return mlflow.models.get_model_info(f"runs:/{run_id}/model").model_uri

def get_model_features(model_uri: str) -> list[str]:
    """
    Get list of models' features' metadata addresses.

    Parameters
    ----------
    model_uri : str
        The URI of the model.

    Returns
    -------
    list[str]
        A list of metadata addresses of model's features.
    
    Raises
    ------
    ValueError
        If the model does not have a 'feature_names' attribute.
    """
    try:
        model_feature_address_path = mltools.model.download_model_artifact(model_uri = model_uri, artifact_name = 'feature_list.yaml')
        model_feature_addresses = yaml.safe_load(open(model_feature_address_path, "r"))
        os.remove(model_feature_address_path)  # clean up downloaded file
    except Exception as e:
        print(f"Error downloading feature_list.yaml for model {model_uri}: {e}")
    
    # add feature to dictionary + set
    return model_feature_addresses if isinstance(model_feature_addresses, list) else [model_feature_addresses]