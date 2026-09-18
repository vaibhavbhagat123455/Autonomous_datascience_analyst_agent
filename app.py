import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

from src.graph import analyst_graph


st.set_page_config(
    page_title="Autonomous Data Science Analyst",
    page_icon="🧠",
    layout="wide",
)


def save_uploaded_file(uploaded_file) -> Path:
    upload_directory = PROJECT_ROOT / "data" / "uploads"
    upload_directory.mkdir(parents=True, exist_ok=True)

    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", uploaded_file.name)
    destination = upload_directory / safe_name
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


st.title("Autonomous Data Science Analyst")
st.caption(
    "Upload a text dataset, describe the task, and receive a trained model "
    "with an integration guide."
)

st.info(
    "Current version: text-based classification datasets. "
    "Your dataset should contain one text column and one target/label column."
)

uploaded_file = st.file_uploader(
    "Upload your dataset",
    type=["csv", "xlsx", "xls"],
)

prompt = st.text_area(
    "Describe what you want the system to do",
    value=(
        "Build a classification model to predict Category from Message. "
        "Compare the models and report accuracy, precision, recall, and F1 score."
    ),
    height=120,
)

if uploaded_file is not None:
    uploaded_path = save_uploaded_file(uploaded_file)

    try:
        if uploaded_path.suffix.lower() == ".csv":
            preview = pd.read_csv(uploaded_path)
        else:
            preview = pd.read_excel(uploaded_path)

        st.subheader("Dataset preview")
        st.write(f"Rows: {preview.shape[0]} | Columns: {preview.shape[1]}")
        st.dataframe(preview.head(10), use_container_width=True)
        st.write("Columns:", list(preview.columns))

        text_columns = preview.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()

        if not text_columns:
            st.error(
                "This version supports text-based datasets only. "
                "The uploaded file has no text column. Image, audio, and vision "
                "datasets are not supported by this project version."
            )
            uploaded_path = None
    except Exception as error:
        st.error(f"Could not preview the uploaded dataset: {error}")
        uploaded_path = None
else:
    uploaded_path = None


if st.button("Run Autonomous Analysis", type="primary"):
    if uploaded_path is None:
        st.error("Upload a CSV or Excel dataset first.")
    elif not prompt.strip():
        st.error("Enter a task prompt first.")
    else:
        initial_state = {
            "dataset_path": str(uploaded_path),
            "user_question": prompt,
            "status": "started",
            "errors": [],
        }

        progress_bar = st.progress(0, text="Preparing workflow...")
        training_bar = st.progress(0, text="Model training has not started yet.")
        current_step = st.empty()
        step_log = st.empty()

        workflow_steps = [
            ("data_profiler", "Data Profiler", 10),
            ("orchestrator", "Gemini Orchestrator", 20),
            ("cleaning_agent", "Cleaning Agent", 35),
            ("eda_agent", "EDA Agent", 50),
            ("model_selector", "Model Selector and Trainer", 80),
            ("explainability_agent", "Explainability Agent", 90),
            ("report_agent", "Report Agent", 100),
        ]
        step_status = {name: "pending" for name, _, _ in workflow_steps}
        step_status[workflow_steps[0][0]] = "running"

        def update_training_progress(fraction: float, message: str) -> None:
            training_bar.progress(
                min(max(fraction, 0.0), 1.0),
                text=f"Model training: {message}",
            )
            progress_bar.progress(
                int(50 + fraction * 30),
                text=f"Overall workflow: {message}",
            )

        def render_step_log() -> None:
            lines = []
            for name, label, _ in workflow_steps:
                icon = {
                    "pending": "⬜",
                    "running": "🔄",
                    "completed": "✅",
                    "failed": "❌",
                }[step_status[name]]
                lines.append(f"{icon} {label}: {step_status[name]}")
            step_log.markdown("\n\n".join(lines))

        render_step_log()
        final_state = dict(initial_state)

        try:
            stream = analyst_graph.stream(
                initial_state,
                context={"progress_callback": update_training_progress},
                stream_mode="updates",
            )

            for update in stream:
                for node_name, node_update in update.items():
                    matching_step = next(
                        (
                            step
                            for step in workflow_steps
                            if step[0] == node_name
                        ),
                        None,
                    )

                    if matching_step:
                        label = matching_step[1]
                        step_status[node_name] = (
                            "failed"
                            if node_update.get("status") == "failed"
                            else "completed"
                        )
                        current_step.info(f"Completed: {label}")
                        progress_bar.progress(
                            matching_step[2],
                            text=f"Completed: {label}",
                        )

                        completed_index = workflow_steps.index(matching_step)
                        if (
                            step_status[node_name] == "completed"
                            and completed_index + 1 < len(workflow_steps)
                        ):
                            next_step = workflow_steps[completed_index + 1]
                            if step_status[next_step[0]] == "pending":
                                step_status[next_step[0]] = "running"
                                current_step.info(
                                    f"Running: {next_step[1]}"
                                )

                    final_state.update(node_update)
                    render_step_log()

        except Exception as error:
            st.error(f"Workflow execution failed unexpectedly: {error}")
            final_state["status"] = "failed"
            final_state["errors"] = [str(error)]

        if final_state.get("errors"):
            st.error("The workflow failed:")
            for error in final_state["errors"]:
                st.write(f"- {error}")
        else:
            st.success(f"Workflow completed: {final_state.get('status')}")

            plan = final_state.get("plan", {})
            model_report = final_state.get("model_report", {})
            metrics = model_report.get("best_model_metrics", {})

            st.subheader("Task plan")
            st.json(plan)

            if metrics:
                st.subheader("Model performance")
                metric_columns = st.columns(4)
                metric_columns[0].metric("Accuracy", f"{metrics.get('accuracy', 0):.2%}")
                metric_columns[1].metric(
                    "Precision", f"{metrics.get('precision_weighted', 0):.2%}"
                )
                metric_columns[2].metric(
                    "Recall", f"{metrics.get('recall_weighted', 0):.2%}"
                )
                metric_columns[3].metric(
                    "F1 score", f"{metrics.get('f1_weighted', 0):.2%}"
                )
                st.write(f"Best model: **{model_report.get('best_model')}**")

            st.subheader("Generated artifacts")
            artifact_paths = {
                "Final HTML report": PROJECT_ROOT / "reports" / "final_report.html",
                "Trained model": PROJECT_ROOT / "reports" / "model" / "best_model.joblib",
                "Standalone test script": PROJECT_ROOT / "reports" / "model" / "test_model.py",
                "Integration guide": PROJECT_ROOT / "reports" / "model" / "integration_guide.md",
                "Feature importance": PROJECT_ROOT / "reports" / "explainability" / "feature_importance.png",
                "Confusion matrix": PROJECT_ROOT / "reports" / "model" / "confusion_matrix.png",
            }

            for label, path in artifact_paths.items():
                if path.exists():
                    mime = "application/octet-stream"
                    if path.suffix == ".html":
                        mime = "text/html"
                    elif path.suffix == ".md":
                        mime = "text/markdown"
                    elif path.suffix == ".py":
                        mime = "text/x-python"
                    elif path.suffix == ".png":
                        mime = "image/png"


                    st.download_button(
                        label=f"Download {label}",
                        data=path.read_bytes(),
                        file_name=path.name,
                        mime=mime,
                        key=f"download_{path.name}",
                    )

            image_paths = [
                PROJECT_ROOT / "reports" / "model" / "confusion_matrix.png",
                PROJECT_ROOT / "reports" / "explainability" / "feature_importance.png",
            ]
            image_columns = st.columns(2)
            for column, image_path in zip(image_columns, image_paths):
                if image_path.exists():
                    column.image(str(image_path), caption=image_path.name)

import json
from src.predictor import predict_message

st.markdown("---")
st.header("🧪 Test Pretrained Model Interactively")
model_path = PROJECT_ROOT / "reports" / "model" / "best_model.joblib"
test_info_path = PROJECT_ROOT / "reports" / "model" / "test_info.json"

if model_path.exists():
    test_info = {}
    if test_info_path.exists():
        try:
            test_info = json.loads(test_info_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    task_title = test_info.get("task_summary", "Trained Model Classifier")
    default_sample = test_info.get(
        "sample_input", "Enter sample text to classify here..."
    )
    custom_code = test_info.get(
        "custom_code",
        """import joblib
import pandas as pd

# Load trained model
model = joblib.load('best_model.joblib')

# Pass text feature in DataFrame
sample_text = "Enter text sample here..."
input_data = pd.DataFrame({'__combined_text__': [sample_text]})

# Get prediction output
prediction = model.predict(input_data)[0]
print("Prediction Result:", prediction)
""",
    )

    st.subheader(f"📌 Task: {task_title}")

    st.markdown("### 💻 Custom Python Testing Code")
    st.caption("Copy and run this code anywhere alongside `best_model.joblib`:")
    st.code(custom_code, language="python")

    st.markdown("### 🎯 Live Model Prediction Sandbox")
    test_input = st.text_area(
        "Enter a test message:",
        value=default_sample,
        height=90,
        key="test_message_input",
    )

    if st.button("Run Test Prediction", type="secondary"):
        if test_input.strip():
            try:
                result = predict_message(test_input, str(model_path))
                st.subheader("Prediction Result")
                col1, col2 = st.columns(2)
                col1.metric("Predicted Label", result.get("prediction", "N/A"))

                if "confidence" in result:
                    col2.metric("Confidence Score", result["confidence"])
                elif "decision_score" in result:
                    col2.metric("Decision Score", result["decision_score"])

                if "class_probabilities" in result:
                    st.write(
                        f"**Class Probabilities:** `{result['class_probabilities']}`"
                    )
            except Exception as err:
                st.error(f"Prediction error: {err}")
        else:
            st.warning("Please enter a message to test.")
else:
    st.info(
        "No trained model found yet (`reports/model/best_model.joblib`). "
        "Run the Autonomous Analysis above to generate the model."
    )


