import pandas as panda
import datetime as dt_time
import logging


def select_report(dfs, selection, description, max_drop_rate=None):
    prev_rows = len(dfs)
    dfs = dfs[selection]
    curr_rows = len(dfs)
    n_dropped = prev_rows - curr_rows

    logging.info("Selection on {}: dropped {} rows."
                 .format(description, n_dropped))

    if max_drop_rate is not None:
        drop_rate = n_dropped / prev_rows
        if drop_rate > max_drop_rate:
            raise RuntimeError("Fraction of rows too large ({}%, "
                               "maximum allowed {}%)."
                               .format(drop_rate * 100, max_drop_rate * 100))

    return dfs


def remove_targets(dfs: panda.DataFrame):
    n_row_original = len(dfs)
    dfs = dfs.dropna()
    logging.info("{} entries had no info on training "
                 "targets and were removed."
                 .format(n_row_original - len(dfs)))
    return dfs


class DtRng:
    def __init__(self, start, end):
        self.start = start
        self.end = end

        if self.end < self.start:
            raise RuntimeError("Start should  defining a "
                               "DtRng")

    @staticmethod
    def dataframe(dfs: panda.DataFrame):
        if 'date' not in dfs:
            raise IndexError("The column 'date' was not found")

        if len(dfs) < 2:
            return None

        return DtRng(dfs.date.min(), dfs.date.max() + dt_time.timedelta(days=1))

    def selects(self, dfs):
        assert 'date' in dfs, "DataFrame dfs should a date column."

        result = dfs.loc[(dfs['date'] >= self.start) &
                         (dfs['date'] < self.end), :].copy()

        if len(result) == 0:
            logging.warning("Date selection causes empty result")

        return result

    def length(self):
        return self.end - self.start

    def __repr__(self):
        return "DtRng: " + str(self.start) + " - " + str(self.end)

    def __and__(self, other):
        r_start = max(self.start, other.start)
        r_end = min(self.end, other.end)

        if r_start < r_end:
            return DtRng(r_start, r_end)
        else:
            return None

    def __eq__(self, other):
        return (self.start == other.start) and (self.end == other.end)
