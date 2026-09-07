import json
import os
from typing import Literal

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field


load_dotenv()


class OrchestratorPlan(BaseModel):
    task_type: Literal[
        "classification",
        "regression",
        "clustering",
        "exploratory_data_analysis",
        "data_quality_analysis",
        "unknown",
    ] = Field(
        description="The main data science task requested by the user."
    )

    target_column: str | None = Field(
        default=None,
        description=(
            "The target column for prediction tasks. "
            "Must be one of the available dataset columns. "
            "Use null when no target column is identifiable."
        ),
    )

    requested_metrics: list[str] = Field(
        description=(
            "Useful metrics or analyses requested or implied by the question."
        )
    )

    recommended_next_agent: Literal[
        "data_profiler",
        "cleaning_agent",
        "eda_agent",
        "model_selector",
        "clarification_required",
    ] = Field(
        description="The next agent that should process the request."
    )

    reasoning: str = Field(
        description="Short explanation for the selected task and next agent."
    )


def create_gemini_model():
    """
    Create the Gemini model used by the Orchestrator Agent.
    """

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY is missing. Add it to the .env file."
        )

    model_name = os.getenv(
        "GEMINI_MODEL",
        "gemini-2.5-flash",
    )

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=0,
        google_api_key=api_key,
    )


def create_orchestrator_plan(
    user_question: str,
    profile: dict,
) -> dict:
    """
    Ask Gemini to create a structured plan using the real dataset schema.
    """

    model = create_gemini_model()

    structured_model = model.with_structured_output(
        OrchestratorPlan,
        method="json_schema",
    )

    available_columns = profile.get("columns", [])

    prompt = f"""
You are the Orchestrator Agent in an autonomous data science system.

Analyze the user's question and the dataset profile.

Rules:
1. Select the most appropriate task_type.
2. target_column must be exactly one of the available columns.
3. If the target column is unclear, use null.
4. Do not invent dataset columns.
5. If the question is ambiguous, choose clarification_required.
6. Recommend the next processing agent.
7. Return only the required structured response.

User question:
{user_question}

Available dataset columns:
{json.dumps(available_columns, indent=2)}

Dataset profile:
{json.dumps(profile, indent=2)}
"""

    response = structured_model.invoke(prompt)

    return response.model_dump()