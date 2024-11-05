from mltools import utils
from mltools.architecture import TF_IDF_Vectorizer_KNN
import mlflow

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
    model_uri = utils.get_nested(config, ['model', 'uri'], None)
    if model_uri is not None:
        wrapped_model = mlflow.pyfunc.load_model(model_uri)
        model = mlflow.pyfunc.PyFuncModel.get_raw_model(wrapped_model)
        return model

    model_type = utils.get_nested(config, ['model', 'type'], None)
    if model_type is None:
        raise ValueError("Model type not provided.")

    model_name = utils.get_nested(config, ['model', 'name'], None)

    if model_type == "TF-IDF-KNN":
        k = utils.get_nested(config, ['parameters', 'k'], 1)
        store_pairwise_distances = utils.get_nested(
            config, ['model', 'parameters', 'store_pairwise_distances'], False)
        model = TF_IDF_Vectorizer_KNN(
            model_name=model_name, k=k, store_pairwise_distances=store_pairwise_distances)
        return model
    elif model_type == "TF-IDF-SVC":
        raise NotImplementedError
    else:
        raise ValueError(f"Model type {model_type} not recognized.")
    return None
