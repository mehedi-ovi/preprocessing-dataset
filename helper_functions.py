import numpy as npy


def error_mean_squared(y_true, y_pred):
    """ Returns  mean squared error  """
    mse = npy.mean(npy.power(y_true - y_pred, 2))
    return mse


def variance_calculation(X):
    """ Return  variance of the features """
    mean = npy.ones(npy.shape(X)) * X.mean(0)
    new_sampls = npy.shape(X)[0]
    variance = (1 / new_sampls) * npy.diag((X - mean).T.dot(X - mean))

    return variance


def standard_dev_calculation(X):
    """ Calculate the standard deviations """
    std_dev = npy.sqrt(variance_calculation(X))
    return std_dev


def euclidian_distance(x1, x2):
    """ Calculates distance between two vectors """
    distance = 0
    for i in range(len(x1)):
        distance += pow((x1[i] - x2[i]), 2)
    return math.sqrt(distance)


def scored_accuracy(y_true, y_pred):
    """ Compare y_true to y_pred and return the accuracy """
    accuracy = npy.sum(y_true == y_pred, axis=0) / len(y_true)
    return accuracy


def covariance_matrix_calculation(X, Y=None):
    """ Calculate the covariance matrix"""
    if Y is None:
        Y = X
    new_sampls = npy.shape(X)[0]
    covariance_matrix = (1 / (new_sampls - 1)) * (X - X.mean(axis=0)).T.dot(Y - Y.mean(axis=0))

    return npy.array(covariance_matrix, dt_timeype=float)
