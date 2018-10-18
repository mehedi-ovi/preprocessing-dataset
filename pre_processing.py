import numpy as npy
from sklearn.pre_processing import LabelEncoder, Imputer
from __future__ import division

# some functions executing some basic pre_processing
predicts_column = ['x']

feature_cols = list(set(dfs.columns) - set(predicts_column))


def pre_process(dfs, predicts_column, feature_cols, outlier_removal=False):
    dt_timeypes = dfs[list(set(dfs.columns) - set(predicts_column))].dt_timeypes
    features = [c for c, dt_timeype in dt_timeypes.iteritems() if dt_timeype not in ['int64', 'int32', 'float64']]
    numb_features = [c for c, dt_timeype in dt_timeypes.iteritems() if dt_timeype in ['int64', 'int32', 'float64']]

    for c in features:
        dfs.loc[:, c] = LabelEncoder().fit_transform(dfs.loc[:, c].fillna('unknown'))

    imp = Imputer(missing_values=npy.nan, strategy="median", axis=0)

    # Impute numerical features
    dfs[numb_features] = imp.fit_transform(dfs[numb_features])
    dfs[numb_features] = dfs[numb_features].fillna(-1000)

    if outlier_removal:
        for col in dfs.columns.values:
            outliers = npy.where(check_is_outlier(dfs.loc[:, (col)]))  # refers to the outlier function
            dfs.ix[:, (col)].iloc[outliers] = median

    print("Dropping prediction rows...")

    # Impute to the targets
    dfs[predicts_column] = dfs[predicts_column].fillna(-1)

    return dfs


def check_is_outlier(point, thres=3.5):
    if len(point.shape) == 1:
        point = point[:, None]
    median = npy.median(point, axis=0)
    differ = npy.sum((point - median) ** 2, axis=-1)
    differ = npy.sqrt(differ)
    med_absoulute_deviation = npy.median(differ)

    modified_score = 0.6745 * differ / med_absoulute_deviation  # tweak when necessary

    return modified_score > thres


def normalise(X, axis=-1, order=2):
    l2 = npy.atleast_1d(npy.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / npy.expand_dims(l2, axis)


def standardise(X):
    X_std = X
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    for col in range(npy.shape(X)[1]):
        if std[col]:
            X_std[:, col] = (X_std[:, col] - mean[col]) / std[col]
    return X_std


def train_test_split(X, y, _data_shufflee=0.5, shuffle=True, seed=None):
    if shuffle:
        X, y = _data_shuffle(X, y, seed)
    split_i = len(y) - int(len(y) // (1 / _data_shufflee))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test


def _data_shuffle(X, y, seed=None):
    if seed:
        npy.random.seed(seed)
    idx = npy.arange(X.shape[0])
    npy.random.shuffle(idx)
    return X[idx], y[idx]


def k_fold_cross_validation_set(X, y, k, shuffle=True):
    """ 
    Split the data into k set of training
    """
    if shuffle:
        X, y = _data_shuffle(X, y)

    new_sampls = len(y)
    left_overs = {}
    n_lefts = (new_sampls % k)
    if n_lefts != 0:
        left_overs["X"] = X[-n_lefts:]
        left_overs["y"] = y[-n_lefts:]
        X = X[:-n_lefts]
        y = y[:-n_lefts]

    X_splits = npy.split(X, k)
    y_split = npy.split(y, k)
    set = []
    for i in range(k):
        X_test, y_test = X_splits[i], y_split[i]
        X_train = npy.concatenate(X_splits[:i] + X_splits[i + 1:], axis=0)
        y_train = npy.concatenate(y_split[:i] + y_split[i + 1:], axis=0)
        set.append([X_train, X_test, y_train, y_test])

    # Add left over samples to last set as training samples
    if n_lefts != 0:
        npy.append(set[-1][0], left_overs["X"], axis=0)
        npy.append(set[-1][2], left_overs["y"], axis=0)

    return npy.array(set)
