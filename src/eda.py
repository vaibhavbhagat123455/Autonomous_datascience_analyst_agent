import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def run_eda(
    cleaned_dataset_path: str,
    output_directory: str = "reports/eda",
    target_column: str | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Run exploratory analysis on the cleaned real dataset."""

    dataframe = pd.read_csv(cleaned_dataset_path)
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    numeric_columns = dataframe.select_dtypes(include="number").columns.tolist()
    categorical_columns = dataframe.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    # Text datasets often have no native numeric columns. Create meaningful
    # numeric features from long text columns for quantitative EDA.
    text_columns = []
    derived_numeric = pd.DataFrame(index=dataframe.index)

    for column in categorical_columns:
        text_values = dataframe[column].fillna("").astype(str)
        average_length = text_values.str.len().mean()

        if average_length >= 20:
            text_columns.append(str(column))
            safe_name = str(column).lower().replace(" ", "_")
            derived_numeric[f"{safe_name}_character_count"] = text_values.str.len()
            derived_numeric[f"{safe_name}_word_count"] = text_values.str.split().str.len()
            derived_numeric[f"{safe_name}_digit_count"] = text_values.str.count(r"\d")
            derived_numeric[f"{safe_name}_exclamation_count"] = text_values.str.count("!")

    artifacts: list[str] = []

    descriptive_statistics = dataframe.describe(
        include="all",
    ).transpose().infer_objects(copy=False).fillna("")

    statistics_path = output_path / "descriptive_statistics.csv"
    descriptive_statistics.to_csv(statistics_path)
    artifacts.append(str(statistics_path))

    plot_dataframe = dataframe[numeric_columns]

    if not numeric_columns and not derived_numeric.empty:
        plot_dataframe = derived_numeric

    plot_columns = plot_dataframe.columns.tolist()
    correlation = pd.DataFrame()

    if plot_columns:
        plt.figure(figsize=(12, 8))
        plot_dataframe.hist(figsize=(12, 8), bins=20)
        plt.suptitle("Numeric Feature Distributions")
        plt.subplots_adjust(hspace=0.6, bottom=0.25)
        distribution_path = output_path / "numeric_distributions.png"
        plt.savefig(distribution_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        artifacts.append(str(distribution_path))

        if len(plot_columns) >= 2:
            plt.figure(figsize=(10, 8))
            correlation = plot_dataframe.corr()
            sns.heatmap(
                correlation,
                annot=True,
                cmap="coolwarm",
                fmt=".2f",
                square=True,
            )
            plt.title("Numeric Feature Correlation Heatmap")
            plt.tight_layout()
            correlation_path = output_path / "correlation_heatmap.png"
            plt.savefig(correlation_path, dpi=150, bbox_inches="tight")
            plt.close("all")
            artifacts.append(str(correlation_path))

    if categorical_columns:
        category_to_plot = categorical_columns[:4]
        figure, axes = plt.subplots(
            len(category_to_plot),
            1,
            figsize=(12, 4 * len(category_to_plot)),
        )

        if len(category_to_plot) == 1:
            axes = [axes]

        for axis, column in zip(axes, category_to_plot):
            counts = dataframe[column].value_counts(dropna=False).head(15)
            counts.plot(kind="bar", ax=axis)
            axis.set_title(f"Top Values: {column}")
            axis.set_xlabel(column)
            axis.set_ylabel("Count")
            axis.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        categorical_path = output_path / "categorical_distributions.png"
        plt.savefig(categorical_path, dpi=150, bbox_inches="tight")
        plt.close("all")
        artifacts.append(str(categorical_path))

    target_analysis: dict[str, Any] = {}

    if target_column and target_column in dataframe.columns:
        target_analysis = {
            "target_column": target_column,
            "target_data_type": str(dataframe[target_column].dtype),
            "target_unique_values": int(dataframe[target_column].nunique()),
            "target_value_counts": {
                str(key): int(value)
                for key, value in dataframe[target_column]
                .value_counts(dropna=False)
                .head(20)
                .items()
            },
        }

    report = {
        "row_count": int(dataframe.shape[0]),
        "column_count": int(dataframe.shape[1]),
        "numeric_columns": [str(column) for column in numeric_columns],
        "derived_text_features": [
            str(column) for column in derived_numeric.columns
        ],
        "text_columns_analyzed": text_columns,
        "categorical_columns": [
            str(column) for column in categorical_columns
        ],
        "target_analysis": target_analysis,
        "correlation_matrix": (
            correlation.round(4).to_dict()
            if not correlation.empty
            else {}
        ),
        "artifacts": artifacts,
    }

    report_path = output_path / "eda_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )
    artifacts.append(str(report_path))

    return report, artifacts
