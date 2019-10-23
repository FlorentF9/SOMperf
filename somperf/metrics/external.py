"""
External indices
"""

import numpy as np
from sklearn.metrics.cluster.supervised import check_clusterings, contingency_matrix
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.metrics import accuracy_score


def class_scatter_index(dist_fun, n_units, y_true, y_pred):
    """Class scatter index (CSI).

    Parameters
    ----------
    dist_fun : function (k : int, l : int) => int
        distance function between units k and l on the map.
    n_units : int
        number of units in the SOM map.
    y_true: array, shape = [n]
        true labels.
    y_pred: array, shape = [n]
        predicted cluster ids.

    Returns
    -------
        csi : float (lower is better)

    References
    ----------
    Elend, L., & Kramer, O. (2019). Self-Organizing Maps with Convolutional Layers.
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    check_clusterings(y_true, y_pred)
    n_classes = y_true.max() + 1
    w = np.zeros((n_classes, n_units))
    for c, k in zip(y_true, y_pred):
        w[c, k] += 1  # w[c, k] = number of c-labeled samples in map cell k
    groups = np.zeros(n_classes, dtype=np.int64)
    for c in range(n_classes):
        # initialize set of remaining units to check
        units_to_check = set(range(n_units))
        while units_to_check:
            k = units_to_check.pop()
            if w[c, k] > 0:  # beginning of a group of units
                groups[c] += 1
                group_units = set()  # collect units that are part of the same group
                for l in units_to_check:
                    if w[c, k] > 0 and w[c, l] > 0 and dist_fun(k, l) == 1:
                        group_units.add(l)
                units_to_check -= group_units
    return np.mean(groups)


def clustering_accuracy(y_true, y_pred):
    """Unsupervised clustering accuracy.

    Can only be used if the number of target classes in y_true is equal to the number of clusters in y_pred.

    Parameters
    ----------
    y_true: array, shape = [n]
        true labels.
    y_pred: array, shape = [n]
        predicted cluster ids.

    Returns
    -------
        accuracy : float in [0,1] (higher is better)
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    check_clusterings(y_true, y_pred)
    w = contingency_matrix(y_true, y_pred)
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def entropy(y_true, y_pred):
    """SOM class distribution entropy measure.

    Parameters
    ----------
    y_true: array, shape = [n]
        true labels.
    y_pred: array, shape = [n]
        predicted cluster ids.

    Returns
    -------
        entropy : float (lower is better)

    References
    ----------
    Elend, L., & Kramer, O. (2019). Self-Organizing Maps with Convolutional Layers.
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    check_clusterings(y_true, y_pred)
    w = contingency_matrix(y_true, y_pred)
    freqs = np.divide(w.max(axis=0), w.sum(axis=0))  # relative frequencies of majority class
    return np.sum(-np.log(freqs))


def normalized_minor_class_occurrence(y_true, y_pred):
    """Normalized minor class occurrence (NMCO).

    Ratio of samples that do not belong to the majority ground-truth label in their cluster. Is equivalent
    to 1 - purity.

    Parameters
    ----------
    y_true: array, shape = [n]
        true labels.
    y_pred: array, shape = [n]
        predicted cluster ids.

    Returns
    -------
        nmco : float in [0,1] (lower is better)

    References
    ----------
    Elend, L., & Kramer, O. (2019). Self-Organizing Maps with Convolutional Layers.
    """
    return 1.0 - purity(y_true, y_pred)


def purity(y_true, y_pred):
    """Clustering purity.

    Parameters
    ----------
    y_true: array, shape = [n]
        true labels.
    y_pred: array, shape = [n]
        predicted cluster ids.

    Returns
    -------
        purity : float in [0,1] (higher is better)
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    check_clusterings(y_true, y_pred)
    w = contingency_matrix(y_true, y_pred)
    label_mapping = w.argmax(axis=0)
    y_pred_voted = np.array([label_mapping[y_pred[i]] for i in range(y_pred.size)])
    return accuracy_score(y_true, y_pred_voted)
