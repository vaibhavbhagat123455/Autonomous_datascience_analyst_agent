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
    img {{ max-width: 100%; border: 1px solid #d1d5db; margin: 10px 0; }}
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
    <img src="model/confusion_matrix.png" alt="Confusion matrix">
  </div>

  <div class="card">
    <h2>Exploratory Analysis</h2>
    <img src="eda/numeric_distributions.png" alt="Numeric distributions">
    <img src="eda/correlation_heatmap.png" alt="Correlation heatmap">
    <img src="eda/categorical_distributions.png" alt="Categorical distributions">
  </div>

  <div class="card">
    <h2>Explainability</h2>
    <img src="explainability/feature_importance.png" alt="Feature importance">
    <p>Detailed feature importance is available in <code>explainability/feature_importance.csv</code>.</p>
  </div>
</body>
</html>
"""

    output_file.write_text(html_report, encoding="utf-8")

    report_data = {
        "output_path": str(output_file),
        "best_model": model_report.get("best_model"),
        "narrative_generated_by": "Google Gemini",
    }

    return str(output_file), report_data
