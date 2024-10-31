from mltools import utils
from mltools.architecture import TF_IDF_Vectorizer_KNN


def setup_model(config):
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
