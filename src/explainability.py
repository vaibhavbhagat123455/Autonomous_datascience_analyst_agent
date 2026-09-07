import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .model_trainer import _prepare_features


def explain_classification_model(
    cleaned_dataset_path: str,
    model_path: str,
    target_column: str,
    output_directory: str = "reports/explainability",
) -> tuple[dict[str, Any], list[str]]:
    """Extract global feature importance from the trained linear model."""

    dataframe = pd.read_csv(cleaned_dataset_path)
    model_pipeline = joblib.load(model_path)

    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' was not found.")

    features, text_columns = _prepare_features(dataframe, target_column)
    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]

    feature_names = preprocessor.get_feature_names_out()
    coefficients = model.coef_
    classes = [str(label) for label in model.classes_]

    rows: list[dict[str, Any]] = []

    if coefficients.shape[0] == 1 and len(classes) == 2:
        coefficient_values = coefficients[0]

        for feature_name, coefficient in zip(feature_names, coefficient_values):
            rows.append(
                {
                    "class": classes[1],
                    "feature": str(feature_name),
                    "coefficient": float(coefficient),
                    "absolute_importance": float(abs(coefficient)),
                }
            )
    else:
        for class_name, class_coefficients in zip(classes, coefficients):
            for feature_name, coefficient in zip(
                feature_names,
                class_coefficients,
            ):
                rows.append(
                    {
                        "class": class_name,
                        "feature": str(feature_name),
                        "coefficient": float(coefficient),
                        "absolute_importance": float(abs(coefficient)),
                    }
                )

    importance_dataframe = pd.DataFrame(rows)
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    importance_file = output_path / "feature_importance.csv"
    importance_dataframe.sort_values(
        "absolute_importance",
        ascending=False,
    ).to_csv(importance_file, index=False)

    chart_rows = []
    for class_name in importance_dataframe["class"].unique():
        class_data = importance_dataframe[
            importance_dataframe["class"] == class_name
        ].sort_values("absolute_importance", ascending=False).head(20)
        chart_rows.append(class_data)

    chart_data = pd.concat(chart_rows, ignore_index=True)
    chart_data["label"] = chart_data["feature"].str.replace(
        "text__combined_text__",
        "",
        regex=False,
    )

    plt.figure(figsize=(12, max(7, len(chart_data) * 0.25)))
    sns.barplot(
        data=chart_data,
        x="coefficient",
        y="label",
        hue="class",
        dodge=False,
        legend=True,
    )
    plt.title("Most Important Model Features")
    plt.xlabel("Coefficient value")
    plt.ylabel("Feature")
    plt.subplots_adjust(left=0.30, right=0.95, top=0.95, bottom=0.15)

    chart_file = output_path / "feature_importance.png"
    plt.savefig(chart_file, dpi=150, bbox_inches="tight")
    plt.close("all")

    top_features = (
        importance_dataframe.sort_values(
            "absolute_importance",
            ascending=False,
        )
        .head(30)
        .to_dict(orient="records")
    )

    report = {
        "target_column": target_column,
        "model_type": type(model).__name__,
        "classes": classes,
        "text_columns": text_columns,
        "explanation_method": "linear_model_coefficients",
        "top_features": top_features,
        "artifacts": [str(importance_file), str(chart_file)],
    }

    report_file = output_path / "explainability_report.json"
    report_file.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    artifacts = [
        str(importance_file),
        str(chart_file),
        str(report_file),
    ]

    return report, artifacts
