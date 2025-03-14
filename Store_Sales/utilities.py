from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
import pandas as pd
import calendar


def grouped_apply_agg(df, group, cols, names, functions):

    grouped = df.sort_values('date').groupby(group, sort=False)
    levels = [i for i in range(len(group))]

    for col in cols:
        for name, function in zip(names, functions):
            df[name + "_" + col] = grouped[col].apply(function).reset_index(level=levels, drop=True)

    return df


def create_scaled_data_by_col(df, min_max_cols, normalize_cols, y_cols, col_name, col):
    if col_name in min_max_cols:
        min_max_cols.remove(col_name)
    if col_name in normalize_cols:
        normalize_cols.remove(col_name)

    db = df[df[col_name] == col]
    db = db.drop(columns=[col_name])

    x_min_max = db[min_max_cols].values.astype(np.float32)
    x_normalize = db[normalize_cols].values.astype(np.float32)
    y = db[y_cols].values.reshape(-2, len(y_cols)).astype(np.float32)

    min_max_scaler = MinMaxScaler().fit(x_min_max)
    normalize_scaler = StandardScaler().fit(x_normalize)
    y_scaler = StandardScaler().fit(y)

    x_min_max = min_max_scaler.transform(x_min_max)
    x_normalize = normalize_scaler.transform(x_normalize)
    y_final = y_scaler.transform(y)

    db[min_max_cols] = x_min_max
    db[normalize_cols] = x_normalize
    db[y_cols] = y_final
    

    return (db, min_max_scaler, normalize_scaler, y_scaler)

def rmsle(y_true, y_pred):
    return np.sqrt(np.mean(np.square(np.log1p(y_true) - np.log1p(y_pred))))


def is_payday(date):
    month_range = calendar.monthrange(date.year, date.month)
    last_day = month_range[1]
    return date.day == 15 or date.day == last_day

