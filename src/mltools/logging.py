# This module contains helper functions that allow easy and consistent logging of experiments to mlflow.

import mlflow.data.dataset_registry
import mltools
import mlflow
import requests
import pandas as pd
import numpy as np
import yaml
import os
import joblib
import uuid
from copy import copy
from sklearn.metrics import classification_report, confusion_matrix, brier_score_loss, RocCurveDisplay
from sklearn.preprocessing import LabelBinarizer
from matplotlib import pyplot as plt
from typing import Literal

def is_remote_mlflow_server_running(verbose: bool = False):
    '''
    Check whether a remote mlflow server uri is set and available
    '''

    try:
        tracking_uri = mlflow.get_tracking_uri()
    except Exception as e:
        if verbose:
            print('No remote mlflow server uri set')
        return False
    
    try:
        response = requests.get(tracking_uri)
    except Exception as e:
        if verbose:
            print('Failed to ping remote mlflow server at uri: ', tracking_uri)
        return False

    if response.status_code == 200:
        if verbose:
            print('Remote mlflow server uri is available')
        return True
    else:
        if verbose:
            print('Remote mlflow server uri is not available')
        return False

def require_run_started():
    '''
    Check whether a run has been started
    '''
    try:
        run_id = mlflow.active_run().info.run_id
    except Exception as e:
        raise Exception('No active mlflow run. Please start a run before logging.')
    return True


#TODO: docs + tests
def create_run(run_name:str, author: str, description: str, task_type: str, project_name: str, experiment_name: str = None, experiment_id: str = None):
    '''
    Experiment may be specified but it has to already exist

    Parameters:
    run_name: str
        Name of the run


    Returns:
    run_id: str
        The id of the run
    '''
    mlflow_server_reachable = is_remote_mlflow_server_running()
    if not mlflow_server_reachable:
        raise Exception('Remote mlflow server not reachable')

    config = {
        "experiment_metadata": {
            "author": author,
            "description": description,
            "task_type": task_type,
            "project_name": project_name
        }
    }

    if experiment_name is not None and experiment_id is not None:
        raise Exception('Specify either experiment_name or experiment_id, but not both.')

    if experiment_name is not None:
        experiment_id = mltools.utils.get_unique_experiment_id_from_name(experiment_name)
    
    mlflow.set_experiment(experiment_id = experiment_id)
    run = mlflow.start_run(run_name = run_name)
    mltools.utils.set_run_tags(config)

    return run.info.run_id

def create_experiment(experiment_name: str):
    '''
    Create an experiment with the given name

    Parameters:
    experiment_name: str
        Name of the experiment

    Returns:
    experiment_id: str
        The id of the experiment
    '''
    mlflow_server_reachable = is_remote_mlflow_server_running()
    if not mlflow_server_reachable:
        raise Exception('Remote mlflow server not reachable')

    experiments = mlflow.search_experiments(filter_string = f'name = "{experiment_name}"')
    if len(experiments) > 0:
        raise Exception('Experiment with name already exists')
    
    experiment_id = mlflow.create_experiment(name=experiment_name, artifact_location=None)

    return experiment_id

def evaluate_classification(model, X_test, y_test, metrics: list = ['classification_report', 'confusion_matrix', 'all_errors', 'most_common_errors', 'roc_auc', 'brier_score', 'confidence_accuracy_curve']):
    '''
    Evaluate a classification model
    '''
    require_run_started()

    # predict_proba
    proba_predictions = model.predict_proba(X_test)
    predictions = model.known_labels[proba_predictions.argmax(axis=1)]
    
    # classification report
    if 'classification_report' in metrics:
        report = pd.DataFrame(classification_report(y_true=y_test,
                                                    y_pred=predictions,
                                                    labels=model.known_labels,
                                                    output_dict=True, zero_division=np.nan,
                                                    )).transpose()
        report['label'] = report.index
        report = report[['label', 'precision',
                        'recall', 'f1-score', 'support']]
        print(report)
        mlflow.log_table(data = report, artifact_file='classification_report.json')

    # cm
    if 'confusion_matrix' in metrics:
        cm = pd.DataFrame(data=confusion_matrix(y_true=y_test, y_pred=predictions, labels=model.known_labels),
            index=model.known_labels, columns=model.known_labels)
        cm_with_index = copy(cm)
        cm_with_index.insert(0, 'True Label', cm.index)
        mlflow.log_table(data=cm_with_index, artifact_file='confusion_matrix.json')

    # most common errors
    if 'most_common_errors' in metrics:
        try:
            cm
        except:
            cm = pd.DataFrame(data=confusion_matrix(y_true=y_test, y_pred=predictions, labels=model.known_labels),
                index=model.known_labels, columns=model.known_labels)

        total_samples = len(predictions)
        unstacked_cm_df = cm.unstack()
        sorted_values_with_indices = unstacked_cm_df.sort_values(
            ascending=False)
        sorted_values_with_indices = sorted_values_with_indices.reset_index()
        sorted_values_with_indices.columns = [
            'Prediction', 'True Label', 'Count']
        sorted_values_with_indices = sorted_values_with_indices[[
            'True Label', 'Prediction', 'Count']]
        sorted_values_with_indices['AccuracyContribution'] = sorted_values_with_indices['Count'] / total_samples
        off_diagonal_nonzero_sorted = sorted_values_with_indices[(sorted_values_with_indices['True Label'] != sorted_values_with_indices['Prediction']) & (
            sorted_values_with_indices['Count'] != 0)].reset_index(drop=True)
        print(off_diagonal_nonzero_sorted)
        mlflow.log_table(
            data=off_diagonal_nonzero_sorted, artifact_file='most_common_errors.json')
        
    # all errors
    if 'all_errors' in metrics:
        X_test_df = pd.DataFrame(X_test)
        error_indices = predictions != y_test
        errors_df = X_test_df[error_indices]
        errors_df['true_label'] = y_test[error_indices]
        errors_df['predicted_label'] = predictions[error_indices]
        errors_df['index'] = errors_df.index
        errors_df = errors_df[['index'] + list(X_test_df.columns) + ['true_label', 'predicted_label']]
        mlflow.log_table(data=errors_df, artifact_file='all_errors.json')
    
    # roc auc curve
    if 'roc_auc' in metrics:
        label_binarizer = LabelBinarizer().fit(model.known_labels)
        true_labels_onehot = label_binarizer.transform(y_test)
        RocCurveDisplay.from_predictions(
            true_labels_onehot.ravel(), proba_predictions.ravel()).plot()
        plt.savefig('roc_curve.png')
        mlflow.log_artifact('roc_curve.png')

    # TODO: pr curve
    # brier score
    if 'brier_score' in metrics:
        brier_score_loss_values = [brier_score_loss(
            y_true=true_labels_onehot[:, i], y_prob=proba_predictions[:, i]) for i in range(len(model.known_labels))]
        brier_score_loss_df = pd.DataFrame(
            data=model.known_labels, columns=['class'])
        brier_score_loss_df['brier_scores'] = brier_score_loss_values
        # add microaveraged brier score
        brier_score_loss_df = pd.concat([brier_score_loss_df, pd.DataFrame(
            {'class': 'micro', 'brier_scores': brier_score_loss(y_true=true_labels_onehot.ravel(), y_prob=proba_predictions.ravel())}, index=[0])], ignore_index=True)
        print(brier_score_loss_df)
        mlflow.log_table(
            data=brier_score_loss_df, artifact_file='brier_score_loss.json')

    # confidence-accuracy curve
        def tradeoff_curve(probas, labels, true_labels):
            predictions = labels[np.argmax(probas, axis=1)]
            correct = true_labels == predictions
            high_confidence_accuracy = []
            manual_ratio = []
            thrs = np.arange(0, 1, 0.01)
            for thr in thrs:
                high_confidence = np.max(probas, axis=1) > thr
                manual = sum(~high_confidence) / len(true_labels)
                acc = sum(high_confidence & correct) / sum(high_confidence)
                manual_ratio.append(manual)
                high_confidence_accuracy.append(acc)
            return thrs, high_confidence_accuracy, manual_ratio

        thresholds, high_confidence_accuracy, manual_ratio = tradeoff_curve(
            proba_predictions, model.known_labels, y_test)
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, high_confidence_accuracy,
                 label="Fraction of Correct High-Confidence Predictions", marker='o', markersize=3)  # Line for y1
        plt.plot(thresholds, manual_ratio, label="Fraction of Low-Confidence Predictions",
                 marker='x', markersize=3)  # Line for y2

        # Add labels, title, and legend
        plt.xlabel('Probability Threshold')
        plt.ylabel('Fraction')
        plt.title('Confidence-Accuracy Curve')
        plt.legend()
        plt.grid(True)  # Optional, for better readability
        plt.gca().xaxis.set_major_locator(plt.MultipleLocator(0.05))
        plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.05))
        plt.savefig('confidence_accuracy_curve.png')
        mlflow.log_artifact('confidence_accuracy_curve.png')
    

def log_model(model, model_name: str, X_example, y_example, pip_requirements: list | str = None, conda_yaml: str = None, servable: bool = True, on_error_signature : Literal['raise','ignore'] = 'raise'):
    '''
    Simplify logging a model to mlflow.

    Parameters:
    model: object
        The model to be logged

    model_name: str
        The name of the model

    X_example: pd.DataFrame
        The training data - used to infer model signature
    
    y_example: pd.Series
        The training labels - used to infer model signature
    
    pip_requirements: list | str
        Package dependencies of the model. Either a list of strings or a path to a requirements.txt-type file.
        Optional.

    conda_yaml: str
        Path to a conda yaml file.
        Optional.
    
    servable: bool
        Whether the model should be logged as a servable model. If True, the model will be wrapped in a PyfuncMlflowWrapper.
        Optional, default is True.
    '''

    require_run_started()

    kwargs = {
        'pip_requirements': pip_requirements,
        'conda_env': conda_yaml
    }
    model_to_log = model if not servable else mltools.architecture.PyfuncMlflowWrapper(model)
    
    mlflow.set_tag('model_id', str(uuid.uuid4()))
    
    if on_error_signature not in ['raise','ignore']:
        raise Exception('on_error_signature must be either "raise" or "ignore"')
    
    try:
        signature = mlflow.models.infer_signature(X_example, y_example, params={'predict_method': 'predict'})
    except Exception as e:
        if on_error_signature == 'ignore':
            signature = None
        elif on_error_signature == 'raise':
            raise Exception('Failed to infer model signature. Please make sure that X_example and y_example are valid inputs for the model. Original error: ' + str(e))

    mlflow.pyfunc.log_model(
        python_model=model_to_log,
        artifact_path='model',
        registered_model_name=model_name,
        signature=signature,
        **kwargs
    )


def log_pandas_dataset(dataset: mlflow.data.pandas_dataset.PandasDataset):
    """
    Parameters
    ----------
    dataset : mlflow.data.PandasDataset
        The dataset to log. The dataset is expected to contain a pandas DataFrame with attrs with at least the following entries: author, description, project_name.

    Returns
    -------
    run_id : str
        The id of the run that was created to log the dataset.

    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    >>> df.attrs['author'] = 'John Doe'
    >>> df.attrs['description'] = 'A dataset with some numbers'
    >>> df.attrs['project_name'] = 'My project'
    >>> dataset = mlflow.data.from_pandas(df, targets='b', name='my_dataset')
    >>> run_id = mltools.logging.log_dataset(dataset)
    """

    if not mltools.logging.is_remote_mlflow_server_running():
        raise Exception('Mlflow server not reachable. Have you set the tracking uri?')
    
    # check for DATASETS experiment, create if necessary
    experiments = mlflow.search_experiments(filter_string = f'name = "DATASETS"')
    if len(experiments) > 1:
        raise Exception('Multiple experiments with name "DATASETS" found')
    if len(experiments) == 0:
        experiment_id = mlflow.create_experiment(name='DATASETS')
    else:
        experiment_id = experiments[0].experiment_id

    # create a run with the name of the dataset. If a run with the same name already exists count them and add 1, resulting in a version to be stored as tag and metadata
    author = dataset.df.attrs['author'] if 'author' in dataset.df.attrs else None
    if not 'description' in dataset.df.attrs:
        raise Exception('Logging a dataset without a description is unacceptable. Shame!')
    description = dataset.df.attrs['description'] 
    project_name = dataset.df.attrs['project_name'] if 'project_name' in dataset.df.attrs else None
    runs_with_this_name = mlflow.search_runs(experiment_ids=experiment_id, filter_string = f'run_name = "{dataset.name}"')
    run_id = mltools.logging.create_run(run_name = dataset.name, author = author, description= description, task_type= 'data_logging', project_name=project_name, experiment_id=experiment_id)
    version = len(runs_with_this_name) + 1
    mlflow.set_tag('version', version)

    # log the metadata and the dataset. If DataSource is not set (or set to CodeDataSource) set its source to the new uri
    mlflow.log_input(dataset)
    temporary_local_filename = f'dataset_{run_id}.joblib'
    joblib.dump(dataset, temporary_local_filename)
    mlflow.log_artifact(local_path=f'dataset_{run_id}.joblib', artifact_path='dataset')
    os.remove(temporary_local_filename)
    
    # create yaml file with metadata for readability
    metadata = {**dataset.df.attrs, 'version': version}
    with open('metadata.yaml', 'w') as file:
        yaml.dump(metadata, file)
    mlflow.log_artifact(local_path = 'metadata.yaml', artifact_path='dataset')
    os.remove('metadata.yaml')

    mlflow.end_run()

    return run_id


