from typing import Any
from typing_extensions import TypedDict


class AnalystState(TypedDict, total=False):
    dataset_path: str
    user_question: str

    profile: dict[str, Any]
    plan: dict[str, Any]
    cleaned_dataset_path: str
    cleaning_report: dict[str, Any]
    eda_report: dict[str, Any]
    eda_artifacts: list[str]
    model_report: dict[str, Any]
    model_artifacts: list[str]
    explainability_report: dict[str, Any]
    explainability_artifacts: list[str]
    final_report_path: str
    final_report_summary: dict[str, Any]
    integration_report_path: str
    status: str
    errors: list[str]


class RunContext(TypedDict, total=False):
    progress_callback: Any
