"""The only data preparation module that reads or writes CSV files."""
import os

import numpy as np
import pandas as pd

from config import config

_NULL_TOKENS = {token.strip().lower() for token in config.NULL_TOKENS}


def _as_missing(value):
    if isinstance(value, str) and value.strip().lower() in _NULL_TOKENS:
        return np.nan
    if isinstance(value, str):
        return value.strip()
    return value


def load_raw_dataset(file_path):
    """Read one raw file as text, guessing nothing; only a null token is missing."""
    frame = pd.read_csv(file_path, dtype=str, keep_default_na=False, na_filter=False)
    for column_name in frame.columns:
        frame[column_name] = frame[column_name].map(_as_missing).astype(object)
    return frame


def save_processed_dataset(dataframe, file_path):
    """Save a cleaned table, creating its folder, with no index column."""
    folder = os.path.dirname(file_path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    dataframe.to_csv(file_path, index=False)


def load_processed_dataset(file_path):
    """Read a cleaned table; types follow the contents and only an empty cell is missing."""
    return pd.read_csv(file_path, keep_default_na=False, na_values=[""], low_memory=False)
