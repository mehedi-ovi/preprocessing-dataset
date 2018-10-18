import pandas as panda


def update_datetime(dfs):
    dfs['datetime_index'] = panda.to_datetime(dfs['datetime_index'])
    dfs = dfs.sort_values('datetime_index', ascending=True)
    dfs = dfs.assign(hour=lambda x: x['datetime_index'].dt_time.hour +
                                    x['datetime_index'].dt_time.minute / 60.)
    dfs = dfs.assign(time_qhour=lambda x: (x['datetime_index'].dt_time.minute / 15).astype(int) + x['hour'] * 4)

    dfs['date'] = dfs['datetime_index'].dt_time.date
    dt_ftrs = panda.DataFrame(dfs['date'].unique(), columns=['date'])
    dt_ftrs = dt_ftrs.assign(year=lambda self: [x.year for x in self['date']])
    dt_ftrs = dt_ftrs.assign(month=lambda self: [x.month for x in self['date']])
    dt_ftrs = dt_ftrs.assign(week=lambda self: [x.isocalendar()[1] for x in self['date']])
    dt_ftrs = dt_ftrs.assign(dayofweek=lambda self: [x.weekday() for x in self['date']])
    dt_ftrs = dt_ftrs.assign(dayofyear=lambda self: [x.timetuple().tm_yday for x in self['date']])

    n_rows_before = dfs.shape[0]
    dfs = panda.merge(dfs, dt_ftrs, on=['date'], suffixes=('_old', ''))

    assert dfs.shape[0] == n_rows_before, "empty rows may be dropped while " \
                                          "merging into date. " \
                                          "This is a error!"

    return dfs
