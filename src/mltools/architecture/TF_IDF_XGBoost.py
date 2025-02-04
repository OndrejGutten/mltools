import mltools
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import xgboost


# TODO: introduce calibrated classifier to the pipeline
# TODO: introduce class weights

class TF_IDF_XGBoost_Classifier():
    def __init__(self, model_name: str, parameters: dict, class_weights: dict = None, cost_matrix : pd.DataFrame = None):
        self.__repr__ = f"TF_IDF_XGB - {model_name}"
        self.name = model_name
        self.parameters = parameters
        self.vectorizer = TfidfVectorizer()

        if class_weights is not None:
            self.set_class_weights(class_weights)
        else:
            self.class_weights = None
        
        if cost_matrix is not None:
            self.set_cost_matrix(cost_matrix)
        else:
            self.cost_matrix = None

        self.trained = False

    def set_class_weights(self, class_weights: dict):
        self.class_weights = class_weights

    def set_cost_matrix(self, cost_matrix: pd.DataFrame):
        mltools.utils.validate_cost_matrix(cost_matrix)
        self.cost_matrix = cost_matrix
    
    def fit(self, X: pd.DataFrame, y: list | pd.Series):
        self.label_encoder = LabelEncoder()
        transformed_y = self.label_encoder.fit_transform(y)

        # find overlapping labels in the train set and the class weights and create a transformed class-weighting dictionary. Any missing labels get the weight of 1
        class_weights_labels = set(self.class_weights.keys()) if self.class_weights is not None else set()
        training_labels = set(self.label_encoder.classes_)
        overlapping_labels = class_weights_labels.intersection(training_labels)
        transformed_overlapping_labels = self.label_encoder.transform(list(overlapping_labels))
        overlapping_labels_weights = [self.class_weights[label] for label in list(overlapping_labels)]
        overlapping_labels_weights_dict = {label: weight for label, weight in zip(transformed_overlapping_labels, overlapping_labels_weights)}
        unweighted_training_labels = training_labels - overlapping_labels
        transformed_unweighted_training_labels = self.label_encoder.transform(list(unweighted_training_labels))
        unweighted_training_labels_dict = {label: 1 for label in transformed_unweighted_training_labels}
        self.training_class_weights_dict = {**overlapping_labels_weights_dict, **unweighted_training_labels_dict}
        self.training_class_weights_array = np.array([self.training_class_weights_dict[label] for label in range(len(self.training_class_weights_dict))])

        # find overlapping labels in the train set and the cost matrix and create a subset-cost matrix. Any missing labels will raise an error
        if self.cost_matrix is not None:
            if not training_labels.issubset(self.cost_matrix.index):
                raise ValueError("Cost matrix does not contain all labels in the training set. Either remove the labels from the training set or provide a cost matrix with all labels.")
            
            subset_cost_matrix = self.cost_matrix.loc[list(training_labels), list(training_labels)].values
            self.custom_loss_function = self._create_cost_matrix_class_weighted_custom_loss_function(subset_cost_matrix, self.training_class_weights_array)
        else:
            self.custom_loss_function = self._create_class_weighted_loss_function(self.training_class_weights_array)

        X_vect = self.vectorizer.fit_transform(X)
        dtrain = xgboost.DMatrix(X_vect, label=transformed_y)

        if not self.parameters.get("num_class"):
            self.parameters["num_class"] = len(np.unique(transformed_y))
        
        if not self.parameters.get('objective'):
            self.parameters['objective'] = "multi:softprob"

        self.xgb = xgboost.train(
            self.parameters,
            dtrain,
            obj=self.custom_loss_function,
        )

        self.trained = True
        self.known_labels = self.label_encoder.inverse_transform(range(len(np.unique(transformed_y))))

    def predict(self, input_texts: str):
        if not self.trained:
            raise Exception("Model not trained")

        X_vect = self.vectorizer.transform(input_texts)
        dtest = xgboost.DMatrix(X_vect)
        argmax_predictions = self.xgb.predict(dtest).argmax(axis=1)
        return self.label_encoder.inverse_transform(argmax_predictions)

    def predict_proba(self, input_texts: str):
        if not self.trained:
            raise Exception("Model not trained")

        X_vect = self.vectorizer.transform(input_texts)
        dtest = xgboost.DMatrix(X_vect)
        return self.xgb.predict(dtest)

    def _create_class_weighted_loss_function(self, class_weights):
        def custom_loss(preds, dtrain):
            labels = dtrain.get_label().astype(int)
            preds = preds.reshape(-1, len(class_weights))
            preds = np.clip(preds, 1e-7, 1 - 1e-7)  # Avoid numerical issues

            # One-hot encoding of labels
            one_hot = np.zeros_like(preds)
            one_hot[np.arange(len(labels)), labels] = 1

            # Weighted log loss
            grad = -class_weights[labels][:, None] * (one_hot - preds)
            hess = class_weights[labels][:, None] * preds * (1 - preds)

            return grad.flatten(), hess.flatten()

        return custom_loss
    
    def _create_cost_matrix_class_weighted_custom_loss_function(self, cost_matrix, class_weights):
        if cost_matrix.shape[0] != cost_matrix.shape[1]:
            raise ValueError("Cost matrix must be a square")
        if len(class_weights) != cost_matrix.shape[0]:
            raise ValueError("Class weights must have the same length as the cost matrix")
        
        def custom_loss(preds,dtrain):
            # Reshape predictions to [n_samples, n_classes]
            num_classes = len(class_weights)
            preds = preds.reshape(-1, num_classes)
            labels = dtrain.get_label().astype(int)

            # Softmax for predicted probabilities
            preds = np.exp(preds) / np.exp(preds).sum(axis=1, keepdims=True)

            # Compute gradients and Hessians
            grad = np.zeros_like(preds)
            hess = np.zeros_like(preds)
            for i in range(len(labels)):
                true_class = labels[i]
                for j in range(num_classes):
                    weight = class_weights[true_class] * (cost_matrix[true_class, j] + 1)
                    grad[i, j] = weight * (preds[i, j] - (j == true_class))
                    hess[i, j] = weight * preds[i, j] * (1 - preds[i, j])

            return grad.flatten(), hess.flatten()

        return custom_loss

