import numpy as np

def classification_cost(y_true, y_predicted, cost_matrix):
    '''
    Calculate the cost of a classification model given the true labels, predicted labels and cost matrix.
    
    Parameters
    ----------
    y_true : array-like
        True labels.
    y_predicted : array-like
        Predicted labels.
    cost_matrix : pandas.DataFrame
        Square cost matrix with matching named index and columns.
        
    Returns
    -------
    dict
        A dictionary with information about the cost of errors of predictions.
        'total_cost': float
            The total cost of the predictions.
        'max_possible_cost': float
            The highest hypothetical cost incurred by worst predictions possible. Depends on y_true and cost_matrix only.
        'min_possible_cost': float
            The lowest hypothetical cost incurred by best predictions possible. Depends on y_true and cost_matrix only.
        'performance': float
            The performance of the model. 1 - (total_cost / max_possible_cost)
    '''
    y_true = np.array(y_true)
    y_predicted = np.array(y_predicted)

    # check if y_true and y_predicted are the same length
    if y_true.shape[0] != y_predicted.shape[0]:
        raise ValueError('y_true and y_predicted must have the same length')

    # check if index and columns match
    if not np.all([col == row for row, col in zip(sorted(cost_matrix.index), sorted(cost_matrix.columns))]):
        raise ValueError('Values in cost_matrix index and columns must match')
        
    # check if all values in y_true and y_predicted are in the cost matrix
    missing_labels_from_y_true = [label for label in np.unique(y_true) if label not in cost_matrix.index]
    if len(missing_labels_from_y_true) > 0:
        raise ValueError(f'All unique values in y_true must be in the cost matrix. Missing values: {missing_labels_from_y_true}')
    
    missing_labels_from_y_predicted = [label for label in np.unique(y_predicted) if label not in cost_matrix.index]
    if len(missing_labels_from_y_predicted) > 0:
        raise ValueError(f'All unique values in y_predicted must be in the cost matrix. Missing values: {missing_labels_from_y_predicted}')

    # calculate total cost
    costs = [cost_matrix.loc[true, predicted] for true, predicted in zip(y_true, y_predicted)]
    total_cost = sum(costs)

    # calculate max possible cost
    max_possible_cost = np.sum([cost_matrix.loc[true_label, :].max() for true_label in y_true])

    # calculate min possible cost
    min_possible_cost = np.sum([cost_matrix.loc[true_label, :].min() for true_label in y_true])

    # calculate performance
    performance = 1 - ((total_cost - min_possible_cost) / max_possible_cost)

    # return dictionary
    return {
        'total_cost': total_cost,
        'max_possible_cost': max_possible_cost,
        'min_possible_cost': min_possible_cost,
        'performance': performance
    }
