import argparse
import json

from .graph import analyst_graph


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Gemini-powered LangGraph analyst."
    )

    parser.add_argument(
        "--file",
        required=True,
        help="Path to a real CSV or Excel dataset.",
    )

    parser.add_argument(
        "--question",
        required=True,
        help="Natural-language question about the dataset.",
    )

    args = parser.parse_args()

    initial_state = {
        "dataset_path": args.file,
        "user_question": args.question,
        "status": "started",
        "errors": [],
    }

    final_state = analyst_graph.invoke(initial_state)

    print("\nLangGraph execution completed.")
    print(f"Status: {final_state.get('status')}")

    if final_state.get("errors"):
        print("\nErrors:")

        for error in final_state["errors"]:
            print(f"- {error}")

        return

    print("\nGemini Orchestrator Plan:")

    print(
        json.dumps(
            final_state.get("plan", {}),
            indent=2,
        )
    )

    if final_state.get("cleaned_dataset_path"):
        print("\nCleaned dataset:")
        print(final_state["cleaned_dataset_path"])

        print("\nCleaning report:")
        print(
            json.dumps(
                final_state.get("cleaning_report", {}),
                indent=2,
            )
        )

    if final_state.get("eda_artifacts"):
        print("\nEDA artifacts:")
        for artifact in final_state["eda_artifacts"]:
            print(artifact)

    if final_state.get("model_report"):
        print("\nModel report:")
        print(
            json.dumps(
                final_state["model_report"],
                indent=2,
            )
        )

    if final_state.get("model_artifacts"):
        print("\nModel artifacts:")
        for artifact in final_state["model_artifacts"]:
            print(artifact)

    if final_state.get("explainability_artifacts"):
        print("\nExplainability artifacts:")
        for artifact in final_state["explainability_artifacts"]:
            print(artifact)

    if final_state.get("final_report_path"):
        print("\nFinal report:")
        print(final_state["final_report_path"])

    if final_state.get("integration_report_path"):
        print("\nIntegration guide:")
        print(final_state["integration_report_path"])


if __name__ == "__main__":
    main()
