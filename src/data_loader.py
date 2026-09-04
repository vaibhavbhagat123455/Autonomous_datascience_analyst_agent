from pathlib import Path
from typing import Any

import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}

def load_dataset(file_path: str, sheet_name: str | int = 0) -> pd.DataFrame:
    """
    Load a real CSV or Excel dataset into a pandas DataFrame.
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS :
        raise ValueError(
            f"Unsupported file type: {path.suffix}."
            f"Supported types are : {SUPPORTED_EXTENSIONS}"
        )

    if path.suffix.lower() == ".csv":
        dataframe = pd.read_csv(path)
    else:
        dataframe = pd.read_excel(path, sheet_name=sheet_name)

    if dataframe.empty:
        raise ValueError("The dataset is empty.")

    return dataframe

def profile_dataset(dataframe: pd.DataFrame) -> dict[str,Any]:
    """
    Generate the basic profile from the loaded real dataset.
    """

    missing_value = dataframe.isna().sum()
    missing_percentage = (
        dataframe.isna().mean().mul(100).round(2)
    )

    column_profiles = {}

    for column in dataframe.columns:
        column_profiles[str (column)] = {
            "data_type" : str(dataframe[column].dtype),
            "row_count": int(dataframe[column].shape[0]),
            "missing_count": int(missing_value[column]),
            "missing_percentage": float(missing_percentage[column]),
            "unique_values": int(dataframe[column].nunique(dropna=True)),
        }

    return {
        "row_count": int(dataframe.shape[0]),
        "column_count": int(dataframe.shape[1]),
        "columns": [str(column) for column in dataframe.columns],
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "column_profiles": column_profiles,
    }
