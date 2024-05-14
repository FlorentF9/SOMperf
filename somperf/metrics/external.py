"""
External indices
"""

import numpy as np
from sklearn.metrics.cluster._supervised import check_clusterings
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics import accuracy_score
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def _contingency_matrix(y_true, y_pred):
    w = np.zeros((y_true.max() + 1, y_pred.max() + 1), dtype=np.int64)
    for c, k in zip(y_true, y_pred):
        w[c, k] += 1  # w[c, k] = number of c-labeled samples in map cell k
    return w


def class_scatter_index(dist_fun, y_true, y_pred):
    """Class scatter index (CSI).

    Parameters
    ----------
    dist_fun : function (k : int, l : int) => int
        distance function between units k and l on the map.
    y_true : array, shape = [n]
        true labels.
    y_pred : array, shape = [n]
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
    n_units = y_pred.max() + 1
    w = _contingency_matrix(y_true, y_pred)
    groups = np.zeros(n_classes, dtype=np.int64)
    for c in range(n_classes):
        connectivity = csr_matrix([[1 if dist_fun(k, l) == 1 else 0
                                   for l in range(n_units) if w[c, l] > 0]
                                   for k in range(n_units) if w[c, k] > 0])
        groups[c] = connected_components(csgraph=connectivity, directed=False, return_labels=False)
    return np.mean(groups)


def clustering_accuracy(y_true, y_pred):
    """Unsupervised clustering accuracy.

    Can only be used if the number of target classes in y_true is equal to the number of clusters in y_pred.

    Parameters
    ----------
    y_true : array, shape = [n]
        true labels.
    y_pred : array, shape = [n]
        predicted cluster ids.

    Returns
    -------
    accuracy : float in [0,1] (higher is better)
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    check_clusterings(y_true, y_pred)
    w = _contingency_matrix(y_true, y_pred).T
    ind = linear_assignment(w.max() - w)
    return np.sum([w[i, j] for i, j in ind]) / y_true.size


def entropy(y_true, y_pred):
    """SOM class distribution entropy measure.

    Parameters
    ----------
    y_true : array, shape = [n]
        true labels.
    y_pred : array, shape = [n]
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
    w = _contingency_matrix(y_true, y_pred)
    freqs = np.divide(w.max(axis=0) + 1e-12, w.sum(axis=0) + 1e-12)  # relative frequencies of majority class
    return np.sum(-np.log(freqs))


def normalized_minor_class_occurrence(y_true, y_pred):
    """Normalized minor class occurrence (NMCO).

    Ratio of samples that do not belong to the majority ground-truth label in their cluster. Is equivalent
    to 1 - purity.

    Parameters
    ----------
    y_true : array, shape = [n]
        true labels.
    y_pred : array, shape = [n]
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
    y_true : array, shape = [n]
        true labels.
    y_pred : array, shape = [n]
        predicted cluster ids.

    Returns
    -------
    purity : float in [0,1] (higher is better)
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    check_clusterings(y_true, y_pred)
    w = _contingency_matrix(y_true, y_pred)
    label_mapping = w.argmax(axis=0)
    y_pred_voted = np.array([label_mapping[y] for y in y_pred])
    return accuracy_score(y_true, y_pred_voted)
