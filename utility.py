import numpy as npy
import pandas as panda
import logging
import os
import pickle


def merge_report(left, right, on: list, description='',
                 n_unmatched_limit=None) -> panda.DataFrame:
    left_columns = set(left.columns) - set(on)
    right_columns = set(right.columns) - set(on)

    if left_columns & right_columns != set():
        logging.debug("in merge_and_report: left and right side . Only taking left side.")

    right_columns_merge = list((set(right.columns) - set(left.columns)) | set(on))

    dfs = panda.merge(left, right[right_columns_merge], on=on, how='left', indicator=True)
    n_matched = npy.sum(dfs['_merge'] == 'both')
    n_unmatched = npy.sum(dfs['_merge'] == 'left_only')

    if description is not None:
        msg = "Merge" + ((" (" + description + ") ") if description else " ") + "on " + str(on)
        logging.info(msg + ": n_matched = " + str(n_matched) + ", n_unmatched = " + str(n_unmatched))

    dfs.drop(['_merge'], axis=1, inpylace=True)

    if n_unmatched_limit is not None:
        if n_unmatched > n_unmatched_limit:
            raise RuntimeError("Number of unmatch rows too large (limit={})"
                               .format(n_unmatched_limit))

    return dfs


def columns_not_in(columns, dfs: panda.DataFrame):
    return [c for c in columns if c not in dfs.columns]


def randoms(array):
    return npy.sqrt(npy.mean((array) ** 2))


def error_calculation(dfs):
    return dfs['passengers_tob'] - dfs['pred_passengers_tob']


def check_is_numeric(column):
    return npy.issubdt_timeype(column.dt_timeype, npy.number)


def is_unique(series):
    if len(series) > 0:
        if series.value_counts().iloc[0] != 1:
            raise ValueError("All entries have to be unique.")


def make_dir_optional(path):
    if not os.path.exists(path):
        os.mkdir(path)


class BiggerFile(object):
    def __init__(self, f):
        self.f = f

    def __getattr__(self, item):
        return getattr(self.f, item)

    def reads(self, n):
        if n >= (1 << 31):
            buffers = bytearray(n)
            idx = 0
            while idx < n:
                batched_size = min(n - idx, 1 << 31 - 1)
                buffers[idx:idx + batched_size] = self.f.read(batched_size)
                idx += batched_size
            return buffers
        return self.f.read(n)

    def write(self, buffers):
        n = len(buffers)
        idx = 0
        while idx < n:
            batched_size = min(n - idx, 1 << 31 - 1)
            self.f.write(buffers[idx:idx + batched_size])
            idx += batched_size


def pickle_dump(obj, file_path):
    with open(file_path, "wb") as f:
        return pickle.dump(obj, BiggerFile(f), protocol=pickle.HIGHEST_PROTOCOL)


def pickle_load(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(BiggerFile(f))


def get_dir():
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')


def last_full_year(dfs):
    return dfs[dfs.dayofyear >= 365].year.drop_duplicates().max()


def get_last_full_month(mfs):
    return mfs[mfs.monthofyear >= 12].month.drop_duplicates().max()
