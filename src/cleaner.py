from pathlib import Path
from typing import Any

import pandas as pd

from .data_loader import load_dataset


def clean_dataset(
    dataset_path: str,
    output_path: str = "data/processed/cleaned_dataset.csv",
    missing_column_threshold: float = 0.40,
) -> tuple[str, dict[str, Any]]:
    """
    Clean a real dataset and save the result.

    Cleaning rules:
    - Trim whitespace from column names and text values.
    - Remove completely empty columns.
    - Remove duplicate rows.
    - Drop non-target columns with more than 40% missing values.
    - Fill numeric missing values with the column median.
    - Fill categorical missing values with the most frequent value,
      or 'Missing' when no mode exists.

    This function does not modify the original dataset.
    """

    dataframe = load_dataset(dataset_path).copy()

    original_rows, original_columns = dataframe.shape

    dataframe.columns = [str(column).strip() for column in dataframe.columns]

    for column in dataframe.select_dtypes(include=["object", "string"]).columns:
        dataframe[column] = dataframe[column].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )

    completely_empty_columns = [
        str(column)
        for column in dataframe.columns
        if dataframe[column].isna().all()
    ]
    dataframe = dataframe.drop(columns=completely_empty_columns)

    duplicate_rows_removed = int(dataframe.duplicated().sum())
    dataframe = dataframe.drop_duplicates().reset_index(drop=True)

    high_missing_columns = [
        str(column)
        for column in dataframe.columns
        if dataframe[column].isna().mean() > missing_column_threshold
    ]
    dataframe = dataframe.drop(columns=high_missing_columns)

    numeric_imputations = {}
    categorical_imputations = {}

    numeric_columns = dataframe.select_dtypes(include="number").columns
    for column in numeric_columns:
        missing_count = int(dataframe[column].isna().sum())

        if missing_count > 0:
            median_value = dataframe[column].median()
            dataframe[column] = dataframe[column].fillna(median_value)
            numeric_imputations[str(column)] = {
                "method": "median",
                "value": float(median_value),
                "count": missing_count,
            }

    categorical_columns = dataframe.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns

    for column in categorical_columns:
        missing_count = int(dataframe[column].isna().sum())

        if missing_count > 0:
            modes = dataframe[column].mode(dropna=True)
            fill_value = modes.iloc[0] if not modes.empty else "Missing"

            dataframe[column] = dataframe[column].fillna(fill_value)
            categorical_imputations[str(column)] = {
                "method": "mode",
                "value": str(fill_value),
                "count": missing_count,
            }

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(destination, index=False)

    report = {
        "original_rows": int(original_rows),
        "original_columns": int(original_columns),
        "final_rows": int(dataframe.shape[0]),
        "final_columns": int(dataframe.shape[1]),
        "duplicate_rows_removed": duplicate_rows_removed,
        "completely_empty_columns_removed": completely_empty_columns,
        "high_missing_columns_removed": high_missing_columns,
        "numeric_imputations": numeric_imputations,
        "categorical_imputations": categorical_imputations,
        "remaining_missing_values": int(dataframe.isna().sum().sum()),
    }

    return str(destination), report
