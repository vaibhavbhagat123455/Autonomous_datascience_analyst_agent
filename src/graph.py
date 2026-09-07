from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from .data_loader import load_dataset, profile_dataset
from .cleaner import clean_dataset
from .eda import run_eda
from .explainability import explain_classification_model
from .model_trainer import train_classification_models
from .report_agent import create_final_report
from .orchestrator import create_orchestrator_plan
from .state import AnalystState, RunContext


def data_profiler_node(state: AnalystState) -> dict[str, Any]:
    """
    Load the real dataset and create its initial profile.
    """

    dataset_path = state.get("dataset_path")

    if not dataset_path:
        return {
            "status": "failed",
            "errors": ["Dataset path was not provided."],
        }

    try:
        dataframe = load_dataset(dataset_path)
        profile = profile_dataset(dataframe)

        return {
            "profile": profile,
            "status": "profile_completed",
            "errors": [],
        }

    except Exception as error:
        return {
            "status": "failed",
            "errors": [str(error)],
        }


def route_after_profiler(
    state: AnalystState,
) -> Literal["orchestrator", "end"]:
    """
    Continue only if profiling succeeded.
    """

    if state.get("status") == "profile_completed":
        return "orchestrator"

    return "end"


def orchestrator_node(state: AnalystState) -> dict[str, Any]:
    """
    Ask Gemini to interpret the user question and create a task plan.
    """

    try:
        user_question = state.get("user_question", "")
        profile = state.get("profile", {})

        if not user_question:
            return {
                "status": "failed",
                "errors": ["User question was not provided."],
            }

        plan = create_orchestrator_plan(
            user_question=user_question,
            profile=profile,
        )

        return {
            "plan": plan,
            "status": "orchestration_completed",
            "errors": [],
        }

    except Exception as error:
        return {
            "status": "failed",
            "errors": [str(error)],
        }


def cleaning_node(state: AnalystState) -> dict[str, Any]:
    """Clean the real dataset and save an auditable output copy."""

    try:
        cleaned_path, cleaning_report = clean_dataset(
            dataset_path=state["dataset_path"],
        )

        return {
            "cleaned_dataset_path": cleaned_path,
            "cleaning_report": cleaning_report,
            "status": "cleaning_completed",
            "errors": [],
        }

    except Exception as error:
        return {
            "status": "failed",
            "errors": [str(error)],
        }


def eda_node(state: AnalystState) -> dict[str, Any]:
    """Analyze the cleaned real dataset and generate visual artifacts."""

    try:
        plan = state.get("plan", {})
        target_column = plan.get("target_column")

        report, artifacts = run_eda(
            cleaned_dataset_path=state["cleaned_dataset_path"],
            target_column=target_column,
        )

        return {
            "eda_report": report,
            "eda_artifacts": artifacts,
            "status": "eda_completed",
            "errors": [],
        }

    except Exception as error:
        return {
            "status": "failed",
            "errors": [str(error)],
        }


def route_after_eda(
    state: AnalystState,
) -> Literal["model_selector", "end"]:
    """Train a model only for a prediction task."""

    plan = state.get("plan", {})

    if (
        state.get("status") == "eda_completed"
        and plan.get("task_type") == "classification"
        and plan.get("target_column")
    ):
        return "model_selector"

    return "end"


def model_selector_node(
    state: AnalystState,
    runtime: Runtime[RunContext],
) -> dict[str, Any]:
    """Select and train a real classification model."""

    try:
        plan = state.get("plan", {})
        target_column = plan.get("target_column")

        if not target_column:
            return {
                "status": "failed",
                "errors": ["A target column is required for classification."],
            }

        context = runtime.context or {}
        progress_callback = context.get("progress_callback")

        report, artifacts = train_classification_models(
            cleaned_dataset_path=state["cleaned_dataset_path"],
            target_column=target_column,
            progress_callback=progress_callback,
        )

        return {
            "model_report": report,
            "model_artifacts": artifacts,
            "status": "model_training_completed",
            "errors": [],
        }

    except Exception as error:
        return {
            "status": "failed",
            "errors": [str(error)],
        }


def explainability_node(state: AnalystState) -> dict[str, Any]:
    """Explain the selected trained model using global feature importance."""

    try:
        plan = state.get("plan", {})
        model_report = state.get("model_report", {})
        target_column = plan.get("target_column")
        model_path = model_report.get("artifacts", [None])[0]

        if not target_column or not model_path:
            return {
                "status": "failed",
                "errors": [
                    "Target column or trained model path is missing for explainability."
                ],
            }

        report, artifacts = explain_classification_model(
            cleaned_dataset_path=state["cleaned_dataset_path"],
            model_path=model_path,
            target_column=target_column,
        )

        return {
            "explainability_report": report,
            "explainability_artifacts": artifacts,
            "status": "explainability_completed",
            "errors": [],
        }

    except Exception as error:
        return {
            "status": "failed",
            "errors": [str(error)],
        }


def report_node(state: AnalystState) -> dict[str, Any]:
    """Create the final HTML report from all completed artifacts."""

    try:
        report_path, report_summary = create_final_report(
            user_question=state.get("user_question", ""),
            cleaning_report=state.get("cleaning_report", {}),
            eda_report=state.get("eda_report", {}),
            model_report=state.get("model_report", {}),
            explainability_report=state.get("explainability_report", {}),
        )

        return {
            "final_report_path": report_path,
            "final_report_summary": report_summary,
            "integration_report_path": report_summary.get("integration_report_path", ""),
            "status": "report_completed",
            "errors": [],
        }

    except Exception as error:
        return {
            "status": "failed",
            "errors": [str(error)],
        }


def route_after_orchestrator(
    state: AnalystState,
) -> Literal["cleaning_agent", "end"]:
    """Pause the workflow when Gemini needs clarification."""

    plan = state.get("plan", {})

    if (
        state.get("status") == "orchestration_completed"
        and plan.get("recommended_next_agent") != "clarification_required"
    ):
        return "cleaning_agent"

    return "end"


def build_graph():
    """
    Build and compile the LangGraph workflow.
    """

    builder = StateGraph(
        AnalystState,
        context_schema=RunContext,
    )

    builder.add_node("data_profiler", data_profiler_node)
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("cleaning_agent", cleaning_node)
    builder.add_node("eda_agent", eda_node)
    builder.add_node("model_selector", model_selector_node)
    builder.add_node("explainability_agent", explainability_node)
    builder.add_node("report_agent", report_node)

    builder.add_edge(START, "data_profiler")

    builder.add_conditional_edges(
        "data_profiler",
        route_after_profiler,
        {
            "orchestrator": "orchestrator",
            "end": END,
        },
    )

    builder.add_conditional_edges(
        "orchestrator",
        route_after_orchestrator,
        {
            "cleaning_agent": "cleaning_agent",
            "end": END,
        },
    )

    builder.add_edge("cleaning_agent", "eda_agent")
    builder.add_conditional_edges(
        "eda_agent",
        route_after_eda,
        {
            "model_selector": "model_selector",
            "end": END,
        },
    )
    builder.add_edge("model_selector", "explainability_agent")
    builder.add_edge("explainability_agent", "report_agent")
    builder.add_edge("report_agent", END)

    return builder.compile()


analyst_graph = build_graph()
