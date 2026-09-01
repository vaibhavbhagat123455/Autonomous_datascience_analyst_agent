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
    