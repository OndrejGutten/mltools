from mltools import utils
from mltools.architecture import TF_IDF_Vectorizer_KNN
import mlflow

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier



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
        #wrapped_model = mlflow.sklearn.load_model(model_uri)
        #model = mlflow.pyfunc.PyFuncModel.get_raw_model(wrapped_model)
        return mlflow.sklearn.load_model(model_uri)
        return model

    model_type = utils.get_nested(config, ['model', 'type'], None)
    if model_type is None:
        raise ValueError("Model type not provided.")

    model_name = utils.get_nested(config, ['model', 'name'], None)
    parameters = utils.get_nested(config,['model','parameters'])

    if model_type == "TF-IDF-KNN":
        model = TF_IDF_Vectorizer_KNN(
            model_name=model_name, **parameters)
        return model
    elif model_type == "TF-IDF-SVM":
        classifier = SVC(probability = True, **parameters)
        model = TF_IDF_Vectorizer_Classifier(classifier = classifier, model_name = model_name)
        return model
    elif model_type == "TF-IDF-RF":
        classifier = RandomForestClassifier(**parameters)
        model = TF_IDF_Vectorizer_Classifier(classifier = classifier, model_name = model_name)
        return model
    else:
        raise ValueError(f"Model type {model_type} not recognized.")
    return None
