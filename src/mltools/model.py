import mltools
import mlflow
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import FunctionTransformer


def load_model(config):
    """
    Load a model and model info from the uri provided in the config.

    Parameters
    ----------
    config : dict
        A dictionary containing the configuration parameters. config['model'] is relevant for this function. It must contain the uri of the model, any other entries will be ignored.

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

    if class_weights_config_dict is not None:
        has_class_weights = True
        class_weights_relative_or_absolute = mltools.utils.get_nested(class_weights_config_dict, ['relative_or_absolute'], 'absolute')
        class_weights_file_path = mltools.utils.get_nested(class_weights_config_dict, ['file_path'], None)
        class_weights = pd.read_csv(class_weights_file_path, header=None)
        class_weights_dict = {str(k):v for k,v in zip(class_weights.iloc[:,0], class_weights.iloc[:,1])}

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
        baseline_classifier = XGBClassifier(**parameters)
        model = mltools.architecture.TF_IDF_XGBoost()
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
    model_uri = utils.get_nested(config, ['model', 'uri'], None)
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