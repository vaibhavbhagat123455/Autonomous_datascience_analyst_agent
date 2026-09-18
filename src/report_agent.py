import html
import json
from pathlib import Path
from typing import Any

from .orchestrator import create_gemini_model


def _generate_narrative(
    user_question: str,
    cleaning_report: dict[str, Any],
    eda_report: dict[str, Any],
    model_report: dict[str, Any],
    explainability_report: dict[str, Any],
) -> str:
    """Ask Gemini for an evidence-based executive summary."""

    prompt = f"""
You are the Report Agent for a real data science project.

Write a concise executive summary using only the evidence below.
Do not invent facts, metrics, columns, or business conclusions.
Mention important limitations, including any excluded target labels.
Use plain text with these sections:

Executive Summary
Key Findings
Recommendations
Limitations

User question:
{user_question}

Cleaning report:
{json.dumps(cleaning_report, indent=2, default=str)}

EDA report:
{json.dumps(eda_report, indent=2, default=str)}

Model report:
{json.dumps(model_report, indent=2, default=str)}

Explainability report:
{json.dumps(explainability_report, indent=2, default=str)}
"""

    response = create_gemini_model().invoke(prompt)
    content = response.content

    if isinstance(content, list):
        return "\n".join(
            item.get("text", str(item)) if isinstance(item, dict) else str(item)
            for item in content
        )

    return str(content)


def _metric_rows(model_report: dict[str, Any]) -> str:
    metrics = model_report.get("best_model_metrics", {})
    rows = []

    for name in [
        "accuracy",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]:
        if name in metrics:
            rows.append(
                "<tr>"
                f"<td>{html.escape(name)}</td>"
                f"<td>{float(metrics[name]):.4f}</td>"
                "</tr>"
            )

    return "".join(rows)


import base64

def _img_tag(image_path: Path, alt_text: str) -> str:
    if image_path.exists():
        data = base64.b64encode(image_path.read_bytes()).decode("utf-8")
        return f'<img src="data:image/png;base64,{data}" alt="{html.escape(alt_text)}">'
    return f'<p><em>Image ({html.escape(alt_text)}) not found.</em></p>'

def create_final_report(
    user_question: str,
    cleaning_report: dict[str, Any],
    eda_report: dict[str, Any],
    model_report: dict[str, Any],
    explainability_report: dict[str, Any],
    output_path: str = "reports/final_report.html",
) -> tuple[str, dict[str, Any]]:
    """Create a self-contained HTML report from the graph outputs."""

    try:
        narrative = _generate_narrative(
            user_question,
            cleaning_report,
            eda_report,
            model_report,
            explainability_report,
        )
    except Exception as error:
        narrative = (
            "Executive Summary\n"
            "The deterministic analysis completed successfully.\n\n"
            "Key Findings\n"
            f"The selected model was {model_report.get('best_model', 'not available')}.\n\n"
            "Recommendations\n"
            "Review the metrics and feature explanations before deployment.\n\n"
            "Limitations\n"
            f"Gemini narrative generation was unavailable: {error}"
        )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    reports_dir = output_file.parent

    cm_img = _img_tag(reports_dir / "model" / "confusion_matrix.png", "Confusion matrix")
    eda_num_img = _img_tag(reports_dir / "eda" / "numeric_distributions.png", "Numeric distributions")
    eda_corr_img = _img_tag(reports_dir / "eda" / "correlation_heatmap.png", "Correlation heatmap")
    eda_cat_img = _img_tag(reports_dir / "eda" / "categorical_distributions.png", "Categorical distributions")
    exp_img = _img_tag(reports_dir / "explainability" / "feature_importance.png", "Feature importance")

    html_report = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Autonomous Data Science Analyst Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto; line-height: 1.6; color: #1f2937; }}
    h1, h2 {{ color: #12366b; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 18px; margin: 18px 0; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #d1d5db; padding: 8px; text-align: left; }}
    th {{ background: #e5eefc; }}
    img {{ max-width: 100%; border: 1px solid #d1d5db; margin: 10px 0; border-radius: 4px; }}
    pre {{ white-space: pre-wrap; background: #f3f4f6; padding: 15px; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>Autonomous Data Science Analyst Report</h1>

  <div class="card">
    <h2>Question</h2>
    <p>{html.escape(user_question)}</p>
  </div>

  <div class="card">
    <h2>Gemini Executive Summary</h2>
    <pre>{html.escape(narrative)}</pre>
  </div>

  <div class="card">
    <h2>Data Preparation</h2>
    <p>Original rows: {cleaning_report.get('original_rows')}</p>
    <p>Final rows: {cleaning_report.get('final_rows')}</p>
    <p>Duplicate rows removed: {cleaning_report.get('duplicate_rows_removed')}</p>
    <p>Remaining missing values: {cleaning_report.get('remaining_missing_values')}</p>
    <p>Excluded rare target classes: {cleaning_report.get('rare_target_classes_excluded', 'See model report')}</p>
  </div>

  <div class="card">
    <h2>Model Performance</h2>
    <p>Best model: <strong>{html.escape(str(model_report.get('best_model', 'N/A')))}</strong></p>
    <table>
      <tr><th>Metric</th><th>Score</th></tr>
      {_metric_rows(model_report)}
    </table>
    <h3>Confusion Matrix</h3>
    {cm_img}
  </div>

  <div class="card">
    <h2>Exploratory Analysis</h2>
    {eda_num_img}
    {eda_corr_img}
    {eda_cat_img}
  </div>

  <div class="card">
    <h2>Explainability</h2>
    {exp_img}
    <p>Detailed feature importance is available in <code>explainability/feature_importance.csv</code>.</p>
  </div>
</body>
</html>
"""

    output_file.write_text(html_report, encoding="utf-8")

    test_info = generate_model_test_info(user_question, model_report)
    test_info_file = output_file.parent / "model" / "test_info.json"
    test_info_file.write_text(json.dumps(test_info, indent=2), encoding="utf-8")

    integration_file = output_file.parent / "model" / "integration_guide.md"
    integration_file.parent.mkdir(parents=True, exist_ok=True)
    integration_file.write_text(
        f"""# Trained Model Integration Guide ({test_info.get('task_summary', 'Model Testing')})

The trained model is `{model_report.get('best_model', 'best_model')}` and is saved as `best_model.joblib`.

## Quick Standalone Testing (`test_model.py`)

A standalone tester script `test_model.py` is saved in the same directory as `best_model.joblib`.

To test custom input messages interactively in your terminal:

```bash
python test_model.py
```

## Installation

```bash
pip install pandas joblib scikit-learn
```

## Load and use the model programmatically

```python
{test_info.get('custom_code', '')}
```

## Sample Input Examples

- **Sample 1**: `{test_info.get('sample_input', '')}`
- **Sample 2**: `{test_info.get('sample_input_2', '')}`

## Important

This model expects text preprocessing via the `__combined_text__` feature column. Use the model for classification matching target column `{model_report.get('target_column')}`.
""",
        encoding="utf-8",
    )

    report_data = {
        "output_path": str(output_file),
        "integration_report_path": str(integration_file),
        "test_info_path": str(test_info_file),
        "best_model": model_report.get("best_model"),
        "narrative_generated_by": "Google Gemini",
    }

    return str(output_file), report_data


def generate_model_test_info(
    user_question: str,
    model_report: dict[str, Any],
) -> dict[str, Any]:
    """Ask Gemini to generate dataset-specific test inputs and code snippet for the trained model."""

    target_col = model_report.get("target_column", "Target")
    text_cols = model_report.get("text_columns", ["text"])
    class_dist = model_report.get("class_distribution", {})
    classes = [str(c) for c in class_dist.keys()]

    prompt = f"""
You are an expert Data Science Agent.
We have trained a machine learning model based on the user request.

User Question / Goal: {user_question}
Target Column: {target_col}
Text Features Used: {text_cols}
Target Output Classes: {classes}

Generate a JSON object with:
1. "task_summary": A short title for this specific task (e.g. "Sentiment Analysis", "Spam Detection", "Topic Classifier").
2. "sample_input": One realistic sample text string for testing this specific model.
3. "sample_input_2": A second realistic sample text string for a different target class label.
4. "custom_code": Clean, ready-to-run Python code snippet showing how to load 'best_model.joblib', format input with '__combined_text__', and print the prediction output for target '{target_col}'.

Return ONLY valid JSON matching this schema:
{{
  "task_summary": "...",
  "sample_input": "...",
  "sample_input_2": "...",
  "custom_code": "..."
}}
"""
    try:
        response = create_gemini_model().invoke(prompt)
        content = response.content
        if isinstance(content, list):
            content = "\n".join(
                str(item.get("text", item)) if isinstance(item, dict) else str(item)
                for item in content
            )
        import re

        match = re.search(r"\{.*\}", str(content), re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception:
        pass

    class_str = ", ".join(classes) if classes else "class labels"
    sample1 = "Enter your test text sample here."
    sample2 = "Enter another sample input here."
    code = f"""import joblib
import pandas as pd

# Load trained model for target: {target_col}
model = joblib.load('best_model.joblib')

# Prepare text input DataFrame
sample_text = "{sample1}"
input_data = pd.DataFrame({{'__combined_text__': [sample_text]}})

# Predict output label ({class_str})
prediction = model.predict(input_data)[0]
print("Predicted {target_col}:", prediction)
"""
    return {
        "task_summary": f"{target_col} Classifier",
        "sample_input": sample1,
        "sample_input_2": sample2,
        "custom_code": code,
    }

