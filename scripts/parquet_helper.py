# parquet_helper.py
# Author: Yolanda Pan
# Date: 12/11/2025
# About: This file contains Utility functions for robust Parquet read/write
# with automatic engine fallback.

import pandas as pd

def read_parquet_any(path: str) -> pd.DataFrame:
    """
    Read a Parquet file using pyarrow, with fallback to fastparquet.
    """
    try:
        return pd.read_parquet(path, engine="pyarrow")
    except Exception:
        return pd.read_parquet(path, engine="fastparquet")


def write_parquet_any(df: pd.DataFrame, path: str) -> None:
    """
    Write a DataFrame to Parquet using pyarrow, with fallback to fastparquet.
    """
    try:
        df.to_parquet(path, engine="pyarrow", index=False)
    except Exception:
        df.to_parquet(path, engine="fastparquet", index=False)
