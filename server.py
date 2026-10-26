import json
import os
import queue
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_file, send_from_directory
from flask_cors import CORS

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

from src.graph import analyst_graph
from src.predictor import predict_message

app = Flask(
    __name__,
    static_folder="web/static",
    static_url_path="/static",
)

# Enable CORS for all domains (allows Vercel frontend to query Render backend)
CORS(app, resources={r"/*": {"origins": "*"}})

reports_dir = PROJECT_ROOT / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)
uploads_dir = PROJECT_ROOT / "data" / "uploads"
uploads_dir.mkdir(parents=True, exist_ok=True)


@app.route("/")
def index():
    index_file = PROJECT_ROOT / "web" / "index.html"
    return send_file(str(index_file))


@app.route("/reports/<path:filename>")
def serve_reports(filename):
    return send_from_directory(str(reports_dir), filename)


@app.route("/api/config", methods=["GET", "POST"])
def config():
    if request.method == "POST":
        payload = request.get_json() or {}
        provider = payload.get("provider", "gemini").lower()
        os.environ["LLM_PROVIDER"] = provider
        return jsonify({"status": "ok", "provider": provider})

    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    model = (
        os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest")
        if provider == "gemini"
        else os.getenv("LOCAL_LLM_MODEL", "qwen3-4b")
    )
    has_gemini_key = bool(os.getenv("GOOGLE_API_KEY"))
    return jsonify({
        "provider": provider,
        "model": model,
        "has_gemini_key": has_gemini_key,
    })


@app.route("/api/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", file.filename)
    dest = uploads_dir / safe_name
    file.save(str(dest))

    try:
        if dest.suffix.lower() == ".csv":
            df = pd.read_csv(dest)
        else:
            df = pd.read_excel(dest)

        rows, cols = df.shape
        columns = list(df.columns)
        text_cols = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
        preview_data = df.head(8).fillna("").to_dict(orient="records")

        return jsonify({
            "status": "success",
            "file_path": str(dest),
            "filename": file.filename,
            "rows": rows,
            "cols": cols,
            "columns": columns,
            "text_columns": text_cols,
            "preview": preview_data,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/run-stream")
def run_pipeline_stream():
    dataset_path = request.args.get("dataset_path", "")
    user_question = request.args.get("user_question", "")

    if not dataset_path or not user_question:
        return jsonify({"error": "Missing parameters"}), 400

    def generate():
        initial_state = {
            "dataset_path": dataset_path,
            "user_question": user_question,
            "status": "started",
            "errors": [],
        }

        workflow_steps = [
            ("data_profiler", "Data Profiler", 15),
            ("orchestrator", "LLM Orchestrator", 30),
            ("cleaning_agent", "Data Sanitization", 45),
            ("eda_agent", "Exploratory Analysis", 60),
            ("model_selector", "Model Trainer", 80),
            ("explainability_agent", "Explainability Engine", 90),
            ("report_agent", "Report Synthesis", 100),
        ]

        yield f"data: {json.dumps({'type': 'init', 'steps': workflow_steps})}\n\n"

        event_queue = queue.Queue()

        def sync_progress_callback(fraction: float, message: str):
            event_queue.put({"type": "progress", "fraction": fraction, "message": message})

        def run_worker():
            try:
                stream = analyst_graph.stream(
                    initial_state,
                    context={"progress_callback": sync_progress_callback},
                    config={"configurable": {"progress_callback": sync_progress_callback}},
                    stream_mode="updates",
                )
                for update in stream:
                    event_queue.put({"type": "node_update", "update": update})
                event_queue.put({"type": "done"})
            except Exception as e:
                event_queue.put({"type": "error", "error": str(e)})

        worker_thread = threading.Thread(target=run_worker)
        worker_thread.daemon = True
        worker_thread.start()

        while True:
            try:
                item = event_queue.get(timeout=1.0)
                if item["type"] == "done":
                    yield f"data: {json.dumps({'type': 'complete'})}\n\n"
                    break
                elif item["type"] == "error":
                    yield f"data: {json.dumps({'type': 'error', 'error': item['error']})}\n\n"
                    break
                else:
                    yield f"data: {json.dumps(item)}\n\n"
            except queue.Empty:
                if not worker_thread.is_alive():
                    break
                yield f": keep-alive\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/reports")
def get_latest_reports():
    model_report_file = reports_dir / "model" / "model_report.json"
    comparison_file = reports_dir / "model" / "model_comparison.json"
    test_info_file = reports_dir / "model" / "test_info.json"
    xai_file = reports_dir / "explainability" / "explainability_report.json"

    data = {
        "has_model": (reports_dir / "model" / "best_model.joblib").exists(),
        "model_report": json.loads(model_report_file.read_text(encoding="utf-8")) if model_report_file.exists() else None,
        "model_comparison": json.loads(comparison_file.read_text(encoding="utf-8")) if comparison_file.exists() else None,
        "test_info": json.loads(test_info_file.read_text(encoding="utf-8")) if test_info_file.exists() else None,
        "explainability": json.loads(xai_file.read_text(encoding="utf-8")) if xai_file.exists() else None,
    }
    return jsonify(data)


@app.route("/api/predict", methods=["POST"])
def run_predict():
    payload = request.get_json() or {}
    message = payload.get("message", "").strip()
    if not message:
        return jsonify({"error": "Message is required"}), 400

    model_path = reports_dir / "model" / "best_model.joblib"
    if not model_path.exists():
        return jsonify({"error": "Trained model not found. Run analysis first."}), 400

    try:
        result = predict_message(message, str(model_path))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/download/<artifact_name>")
def download_artifact(artifact_name):
    file_map = {
        "final_report": reports_dir / "final_report.html",
        "best_model": reports_dir / "model" / "best_model.joblib",
        "test_model": reports_dir / "model" / "test_model.py",
        "integration_guide": reports_dir / "model" / "integration_guide.md",
        "feature_importance_csv": reports_dir / "explainability" / "feature_importance.csv",
        "confusion_matrix": reports_dir / "model" / "confusion_matrix.png",
        "cleaned_dataset": PROJECT_ROOT / "data" / "processed" / "cleaned_dataset.csv",
    }
    if artifact_name not in file_map or not file_map[artifact_name].exists():
        return jsonify({"error": "Artifact not found"}), 404

    file_path = file_map[artifact_name]
    return send_file(str(file_path), as_attachment=True, download_name=file_path.name)


if __name__ == "__main__":
    print("🚀 Launching Autonomous Data Science Analyst Agent Web UI...")
    print("📍 URL: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)
