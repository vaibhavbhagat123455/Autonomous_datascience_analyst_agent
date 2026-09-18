import json
from pathlib import Path
from typing import Any, Callable

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def _prepare_features(
    dataframe: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, list[str]]:
    """Prepare text, categorical, and numeric columns without using the target."""

    features = dataframe.drop(columns=[target_column]).copy()
    text_columns: list[str] = []

    for column in features.select_dtypes(
        include=["object", "string"]
    ).columns:
        average_length = features[column].fillna("").astype(str).str.len().mean()
        if average_length >= 20:
            text_columns.append(str(column))

    if text_columns:
        features["__combined_text__"] = (
            features[text_columns]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
        )
        features = features.drop(columns=text_columns)

    return features, text_columns


def train_classification_models(
    cleaned_dataset_path: str,
    target_column: str,
    output_directory: str = "reports/model",
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Compare real classification models and save the best trained model."""

    dataframe = pd.read_csv(cleaned_dataset_path)

    if target_column not in dataframe.columns:
        raise ValueError(
            f"Target column '{target_column}' was not found. "
            f"Available columns: {list(dataframe.columns)}"
        )

    if dataframe[target_column].nunique() < 2:
        raise ValueError("Classification requires at least two target classes.")

    dataframe = dataframe.dropna(subset=[target_column]).copy()
    features, text_columns = _prepare_features(dataframe, target_column)
    target = dataframe[target_column].astype(str)

    class_counts = target.value_counts()
    rare_target_classes = [
        str(label) for label in class_counts[class_counts < 2].index
    ]

    if rare_target_classes:
        valid_rows = ~target.isin(rare_target_classes)
        features = features.loc[valid_rows]
        target = target.loc[valid_rows]

    if target.nunique() < 2:
        raise ValueError(
            "After removing target classes with fewer than two rows, "
            "fewer than two classes remain. Check the target column."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )

    text_feature_columns = ["__combined_text__"] if text_columns else []
    categorical_columns = features.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()
    numeric_columns = features.select_dtypes(include="number").columns.tolist()

    transformers = []

    if text_feature_columns:
        transformers.append(
            (
                "text",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=20_000,
                ),
                "__combined_text__",
            )
        )

    if categorical_columns:
        from sklearn.preprocessing import OneHotEncoder

        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_columns,
            )
        )

    if numeric_columns:
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_columns,
            )
        )

    if not transformers:
        raise ValueError("No usable feature columns were found.")

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    model_candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=1_000,
            class_weight="balanced",
            random_state=42,
        ),
        "linear_svm": LinearSVC(
            class_weight="balanced",
            random_state=42,
        ),
    }

    evaluation_results: dict[str, Any] = {}
    trained_pipelines: dict[str, Pipeline] = {}

    total_models = len(model_candidates)

    for model_index, (model_name, estimator) in enumerate(
        model_candidates.items(),
        start=1,
    ):
        if progress_callback:
            progress_callback(
                (model_index - 1) / total_models,
                f"Training {model_name.replace('_', ' ').title()}...",
            )

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    ColumnTransformer(transformers=transformers),
                ),
                ("model", estimator),
            ]
        )

        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)

        evaluation_results[model_name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_weighted": float(
                precision_score(y_test, predictions, average="weighted", zero_division=0)
            ),
            "recall_weighted": float(
                recall_score(y_test, predictions, average="weighted", zero_division=0)
            ),
            "f1_weighted": float(
                f1_score(y_test, predictions, average="weighted", zero_division=0)
            ),
            "classification_report": classification_report(
                y_test,
                predictions,
                output_dict=True,
                zero_division=0,
            ),
        }
        trained_pipelines[model_name] = pipeline

        if progress_callback:
            progress_callback(
                model_index / total_models,
                f"Completed {model_name.replace('_', ' ').title()}.",
            )

    best_model_name = max(
        evaluation_results,
        key=lambda name: evaluation_results[name]["f1_weighted"],
    )
    best_pipeline = trained_pipelines[best_model_name]
    best_predictions = best_pipeline.predict(x_test)

    model_file = output_path / "best_model.joblib"
    joblib.dump(best_pipeline, model_file)

    comparison_file = output_path / "model_comparison.json"
    comparison_file.write_text(
        json.dumps(evaluation_results, indent=2, default=str),
        encoding="utf-8",
    )

    labels = sorted(target.unique())
    matrix = confusion_matrix(y_test, best_predictions, labels=labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(f"Confusion Matrix: {best_model_name}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()

    confusion_file = output_path / "confusion_matrix.png"
    plt.savefig(confusion_file, dpi=150, bbox_inches="tight")
    plt.close("all")

    test_script_file = output_path / "test_model.py"
    test_script_content = '''"""
Standalone Model Tester Script
Run this script to test custom inputs against the trained model in this folder.
"""
import json
from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path(__file__).parent / "best_model.joblib"
INFO_PATH = Path(__file__).parent / "test_info.json"

def run_test():
    if not MODEL_PATH.exists():
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    print(f"Loading trained model from {MODEL_PATH.name}...")
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!\\n")

    test_info = {}
    if INFO_PATH.exists():
        try:
            test_info = json.loads(INFO_PATH.read_text(encoding="utf-8"))
            print(f"Task: {test_info.get('task_summary', 'Model Classification')}")
        except Exception:
            pass

    samples = []
    if "sample_input" in test_info:
        samples.append(test_info["sample_input"])
    if "sample_input_2" in test_info:
        samples.append(test_info["sample_input_2"])

    if samples:
        print("--- Preset Sample Predictions ---")
        for sample in samples:
            input_df = pd.DataFrame({"__combined_text__": [sample]})
            pred = model.predict(input_df)[0]
            score_str = ""
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)[0]
                score_str = f" (Confidence: {max(probs):.2%})"
            elif hasattr(model, "decision_function"):
                score = model.decision_function(input_df)[0]
                score_str = f" (Decision Score: {score:.4f})"
            print(f"Input: '{sample}'")
            print(f"--> Predicted Output: [{pred}]{score_str}\\n")

    print("=" * 60)
    print("Enter a text message to test model prediction.")
    print("Type 'exit', 'quit', or press Ctrl+C to stop.")
    print("=" * 60 + "\\n")

    while True:
        try:
            text = input("Enter message: ").strip()
            if text.lower() in ["exit", "quit", "q"]:
                print("Exiting model tester.")
                break
            if not text:
                continue

            input_df = pd.DataFrame({"__combined_text__": [text]})
            prediction = model.predict(input_df)[0]

            confidence = ""
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)[0]
                confidence = f" (Confidence: {max(probs):.2%})"
            elif hasattr(model, "decision_function"):
                score = model.decision_function(input_df)[0]
                confidence = f" (Decision Score: {score:.4f})"

            print(f"--> Prediction: [{prediction}]{confidence}\\n")
        except (KeyboardInterrupt, EOFError):
            print("\\nExiting model tester.")
            break

if __name__ == "__main__":
    run_test()
'''
    test_script_file.write_text(test_script_content, encoding="utf-8")


    report = {
        "task_type": "classification",
        "target_column": target_column,
        "text_columns": text_columns,
        "rare_target_classes_excluded": rare_target_classes,
        "training_rows": int(len(x_train)),
        "testing_rows": int(len(x_test)),
        "class_distribution": {
            str(key): int(value) for key, value in target.value_counts().items()
        },
        "candidate_models": list(model_candidates.keys()),
        "best_model": best_model_name,
        "best_model_metrics": evaluation_results[best_model_name],
        "artifacts": [
            str(model_file),
            str(comparison_file),
            str(confusion_file),
            str(test_script_file),
        ],
    }

    report_file = output_path / "model_report.json"
    report_file.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    artifacts = [
        str(model_file),
        str(comparison_file),
        str(confusion_file),
        str(test_script_file),
        str(report_file),
    ]

    return report, artifacts

