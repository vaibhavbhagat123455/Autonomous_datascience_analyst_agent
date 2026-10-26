import json
import os
import re
import time
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

from src.graph import analyst_graph
from src.predictor import predict_message

# ──────────────────────────────────────────────
#  PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Data Science Analyst Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
#  GLOBAL CSS
# ──────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Inter font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── background ── */
    .stApp { background: #0f1117; }

    /* ── sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d27 0%, #12151f 100%);
        border-right: 1px solid #2a2d3e;
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }

    /* ── hero banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #1e3a5f 0%, #0d1b2e 50%, #1a0533 100%);
        border: 1px solid #2d4a6b;
        border-radius: 16px;
        padding: 2.4rem 2.8rem;
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute; top: -60px; right: -60px;
        width: 220px; height: 220px;
        background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero-title {
        font-size: 2.1rem; font-weight: 800;
        background: linear-gradient(90deg, #60a5fa, #a78bfa, #f472b6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0 0 0.4rem 0; line-height: 1.2;
    }
    .hero-sub {
        color: #94a3b8; font-size: 1rem; font-weight: 400; margin: 0;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(99,102,241,0.18); border: 1px solid rgba(99,102,241,0.4);
        color: #a5b4fc; font-size: 0.72rem; font-weight: 600;
        padding: 3px 10px; border-radius: 20px; margin-right: 6px;
        letter-spacing: 0.05em; text-transform: uppercase;
    }

    /* ── glass card ── */
    .glass-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
    }
    .card-title {
        font-size: 0.82rem; font-weight: 600; letter-spacing: 0.08em;
        text-transform: uppercase; color: #64748b; margin-bottom: 0.9rem;
    }

    /* ── step cards ── */
    .step-item {
        display: flex; align-items: center; gap: 12px;
        padding: 9px 14px; border-radius: 10px;
        margin-bottom: 7px; transition: background 0.2s;
    }
    .step-pending  { background: rgba(100,116,139,0.07); }
    .step-running  { background: rgba(250,204,21,0.10); border: 1px solid rgba(250,204,21,0.25); }
    .step-completed{ background: rgba(34,197,94,0.09);  border: 1px solid rgba(34,197,94,0.22); }
    .step-failed   { background: rgba(239,68,68,0.10);  border: 1px solid rgba(239,68,68,0.25); }

    .step-icon  { font-size: 1.1rem; flex-shrink: 0; }
    .step-label { font-size: 0.87rem; font-weight: 500; color: #e2e8f0; }
    .step-status-txt {
        font-size: 0.73rem; font-weight: 600; margin-left: auto;
        letter-spacing: 0.04em; text-transform: uppercase;
    }
    .txt-pending   { color: #64748b; }
    .txt-running   { color: #facc15; }
    .txt-completed { color: #4ade80; }
    .txt-failed    { color: #f87171; }

    /* ── metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.12), rgba(168,85,247,0.08));
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 12px; padding: 1.1rem 1rem;
        text-align: center;
    }
    .metric-label { font-size: 0.72rem; font-weight: 600; color: #64748b; letter-spacing: 0.06em; text-transform: uppercase; }
    .metric-value { font-size: 1.9rem; font-weight: 800; color: #a5b4fc; line-height: 1.1; margin: 4px 0; }
    .metric-sub   { font-size: 0.72rem; color: #94a3b8; }

    /* ── upload area ── */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.03) !important;
        border: 2px dashed rgba(99,102,241,0.35) !important;
        border-radius: 12px !important;
    }

    /* ── text area / input ── */
    textarea, [data-baseweb="textarea"] {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(99,102,241,0.3) !important;
        border-radius: 10px !important; color: #e2e8f0 !important;
    }

    /* ── buttons ── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        border: none !important; border-radius: 10px !important;
        font-weight: 700 !important; font-size: 0.95rem !important;
        padding: 0.65rem 2rem !important; color: white !important;
        box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
        transition: all 0.2s !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 28px rgba(99,102,241,0.55) !important;
    }
    .stButton > button[kind="secondary"] {
        background: rgba(99,102,241,0.12) !important;
        border: 1px solid rgba(99,102,241,0.4) !important;
        border-radius: 10px !important; color: #a5b4fc !important;
        font-weight: 600 !important;
    }

    /* ── download buttons ── */
    .stDownloadButton > button {
        background: rgba(34,197,94,0.1) !important;
        border: 1px solid rgba(34,197,94,0.3) !important;
        border-radius: 8px !important; color: #4ade80 !important;
        font-weight: 600 !important; width: 100% !important;
    }

    /* ── dataframe ── */
    [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

    /* ── code blocks ── */
    .stCode { border-radius: 10px !important; }

    /* ── divider ── */
    hr { border-color: rgba(255,255,255,0.07) !important; margin: 1.8rem 0 !important; }

    /* ── section header ── */
    .section-header {
        display: flex; align-items: center; gap: 10px;
        border-left: 3px solid #4f46e5;
        padding-left: 12px; margin-bottom: 1.2rem;
    }
    .section-header h3 { margin: 0; font-size: 1.1rem; font-weight: 700; color: #e2e8f0; }

    /* ── prediction badge ── */
    .pred-ham  { background: rgba(34,197,94,0.15);  border: 1px solid rgba(34,197,94,0.4);  color: #4ade80;  }
    .pred-spam { background: rgba(239,68,68,0.15);  border: 1px solid rgba(239,68,68,0.4);  color: #f87171;  }
    .pred-box  {
        border-radius: 12px; padding: 1.2rem 1.5rem;
        font-size: 1.5rem; font-weight: 800; text-align: center;
    }

    /* ── scrollbar ── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #12151f; }
    ::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────
def save_uploaded_file(uploaded_file) -> Path:
    upload_dir = PROJECT_ROOT / "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", uploaded_file.name)
    dest = upload_dir / safe_name
    dest.write_bytes(uploaded_file.getbuffer())
    return dest


def render_step_list(step_status: dict, workflow_steps: list):
    icons = {"pending": "○", "running": "◉", "completed": "✓", "failed": "✕"}
    html = ""
    for name, label, *_ in workflow_steps:
        state = step_status.get(name, "pending")
        html += (
            f'<div class="step-item step-{state}">'
            f'  <span class="step-icon">{icons[state]}</span>'
            f'  <span class="step-label">{label}</span>'
            f'  <span class="step-status-txt txt-{state}">{state}</span>'
            f"</div>"
        )
    return html


def metric_card(label: str, value: str, sub: str = ""):
    return (
        f'<div class="metric-card">'
        f'  <div class="metric-label">{label}</div>'
        f'  <div class="metric-value">{value}</div>'
        f'  <div class="metric-sub">{sub}</div>'
        f"</div>"
    )


# ──────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding: 1rem 0 0.5rem;">
            <div style="font-size:2.8rem;">🤖</div>
            <div style="font-size:1.05rem; font-weight:700; color:#a5b4fc; margin-top:6px;">
                DS Analyst Agent
            </div>
            <div style="font-size:0.75rem; color:#64748b; margin-top:2px;">
                Powered by LangGraph + LLM
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("##### 🗂 Navigation")
    page = st.radio(
        "",
        ["🚀  Analyse Dataset", "🧪  Prediction Sandbox", "📊  Model Reports", "ℹ️  About"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("##### ⚙️ Configuration")

    llm_provider = st.selectbox(
        "LLM Provider",
        ["Local (Qwen3-14B)", "Cloud (Gemini 2.5 Flash)"],
        help="Local = 100% private, no data leaves your machine.",
    )

    st.markdown("---")
    st.markdown(
        """
        <div style='font-size:0.72rem; color:#475569; line-height:1.6;'>
        <b style='color:#64748b;'>Stack</b><br>
        🔗 LangGraph · LangChain<br>
        🧠 Qwen3-14B / Gemini<br>
        📦 Scikit-Learn · Pandas<br>
        📊 Matplotlib · Seaborn<br>
        🌐 Streamlit 1.63
        </div>
        """,
        unsafe_allow_html=True,
    )

page_name = page.split("  ", 1)[-1].strip()

# ──────────────────────────────────────────────
#  PAGE: ANALYSE DATASET
# ──────────────────────────────────────────────
if page_name == "Analyse Dataset":

    # ── Hero ──
    st.markdown(
        """
        <div class="hero-banner">
            <div style="margin-bottom:0.8rem;">
                <span class="hero-badge">AI Agent</span>
                <span class="hero-badge">AutoML</span>
                <span class="hero-badge">XAI</span>
            </div>
            <p class="hero-title">Autonomous Data Science Analyst</p>
            <p class="hero-sub">
                Upload a dataset → describe your goal → get a trained model,<br>
                EDA visuals, explainability report and a prediction sandbox — all in seconds.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Input columns ──
    col_upload, col_prompt = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<div class="card-title">📁 Upload Dataset</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Drop your CSV or Excel file here",
            type=["csv", "xlsx", "xls"],
            label_visibility="collapsed",
        )

        uploaded_path = None
        if uploaded_file:
            uploaded_path = save_uploaded_file(uploaded_file)
            try:
                preview = (
                    pd.read_csv(uploaded_path)
                    if uploaded_path.suffix.lower() == ".csv"
                    else pd.read_excel(uploaded_path)
                )
                text_cols = preview.select_dtypes(include=["object", "string", "category"]).columns.tolist()
                if not text_cols:
                    st.error("No text columns found. Please upload a text-based dataset.")
                    uploaded_path = None
                else:
                    st.success(f"✅  **{uploaded_file.name}** loaded — {preview.shape[0]:,} rows × {preview.shape[1]} cols")
                    with st.expander("👁  Preview first 8 rows", expanded=False):
                        st.dataframe(preview.head(8), use_container_width=True)
                        cols_str = "  `" + "`  `".join(preview.columns.tolist()) + "`"
                        st.markdown(f"**Columns:** {cols_str}")
            except Exception as e:
                st.error(f"Could not read file: {e}")
                uploaded_path = None

    with col_prompt:
        st.markdown('<div class="card-title">✏️ Task Description</div>', unsafe_allow_html=True)
        prompt = st.text_area(
            "Describe your goal",
            value=(
                "Build a classification model to predict Category from Message. "
                "Compare the models and report accuracy, precision, recall, and F1 score."
            ),
            height=138,
            label_visibility="collapsed",
            placeholder="e.g. Build a spam classifier for the Message column and explain what words indicate spam.",
        )
        st.caption("💡 Be specific about your target column and what metrics you care about.")

    st.markdown("<br>", unsafe_allow_html=True)

    run_col, _ = st.columns([1, 3])
    with run_col:
        run_btn = st.button("⚡  Run Autonomous Analysis", type="primary", use_container_width=True)

    # ── Workflow Execution ──
    if run_btn:
        if not uploaded_path:
            st.error("Please upload a valid dataset first.")
            st.stop()
        if not prompt.strip():
            st.error("Please enter a task description.")
            st.stop()

        st.markdown("---")
        st.markdown(
            '<div class="section-header"><h3>🔄 Pipeline Execution</h3></div>',
            unsafe_allow_html=True,
        )

        exec_left, exec_right = st.columns([1, 1], gap="large")

        with exec_left:
            st.markdown('<div class="card-title">Agent Progress</div>', unsafe_allow_html=True)
            steps_placeholder = st.empty()
            overall_bar = st.progress(0, text="Initialising pipeline…")

        with exec_right:
            st.markdown('<div class="card-title">Model Training</div>', unsafe_allow_html=True)
            training_bar = st.progress(0, text="Waiting for model training…")
            status_box = st.empty()

        workflow_steps = [
            ("data_profiler",        "Data Profiler",             5,  15),
            ("orchestrator",         "LLM Orchestrator",         15,  30),
            ("cleaning_agent",       "Data Cleaning Agent",      30,  45),
            ("eda_agent",            "EDA & Visualisation",      45,  60),
            ("model_selector",       "Model Selector & Trainer", 60,  80),
            ("explainability_agent", "Explainability Agent",     80,  90),
            ("report_agent",         "Report Synthesis",         90, 100),
        ]
        step_status = {n: "pending" for n, *_ in workflow_steps}
        step_status[workflow_steps[0][0]] = "running"

        def refresh_steps():
            steps_placeholder.markdown(
                render_step_list(step_status, workflow_steps),
                unsafe_allow_html=True,
            )

        refresh_steps()

        def update_training(fraction: float, message: str):
            training_bar.progress(
                min(max(float(fraction), 0.0), 1.0),
                text=f"Model training — {message}",
            )
            overall_bar.progress(
                int(60 + fraction * 20),
                text=f"Overall — {message}",
            )

        initial_state = {
            "dataset_path": str(uploaded_path),
            "user_question": prompt,
            "status": "started",
            "errors": [],
        }
        final_state = dict(initial_state)

        try:
            stream = analyst_graph.stream(
                initial_state,
                context={"progress_callback": update_training},
                config={"configurable": {"progress_callback": update_training}},
                stream_mode="updates",
            )

            for update in stream:
                for node_name, node_update in update.items():
                    match = next((s for s in workflow_steps if s[0] == node_name), None)
                    if match:
                        idx = workflow_steps.index(match)
                        step_status[node_name] = (
                            "failed" if node_update.get("status") == "failed" else "completed"
                        )
                        overall_bar.progress(match[3], text=f"Completed — {match[1]}")
                        if step_status[node_name] == "completed" and idx + 1 < len(workflow_steps):
                            nxt = workflow_steps[idx + 1]
                            if step_status[nxt[0]] == "pending":
                                step_status[nxt[0]] = "running"
                                overall_bar.progress(nxt[2], text=f"Running — {nxt[1]}…")
                    final_state.update(node_update)
                    refresh_steps()

        except Exception as err:
            st.error(f"Pipeline error: {err}")
            final_state["status"] = "failed"
            final_state["errors"] = [str(err)]

        # ── Results ──
        if final_state.get("errors"):
            st.error("Pipeline failed:")
            for e in final_state["errors"]:
                st.write(f"- {e}")
        else:
            overall_bar.progress(100, text="✅ Analysis complete!")
            status_box.success("All agents completed successfully.")

            model_report = final_state.get("model_report", {})
            metrics = model_report.get("best_model_metrics", {})

            if metrics:
                st.markdown("---")
                st.markdown(
                    '<div class="section-header"><h3>📈 Model Performance</h3></div>',
                    unsafe_allow_html=True,
                )
                m1, m2, m3, m4 = st.columns(4)
                for col, label, key, sub in [
                    (m1, "Accuracy",  "accuracy",           "Overall correct"),
                    (m2, "Precision", "precision_weighted",  "Weighted avg"),
                    (m3, "Recall",    "recall_weighted",     "Weighted avg"),
                    (m4, "F1 Score",  "f1_weighted",         "Weighted avg"),
                ]:
                    val = metrics.get(key, 0)
                    col.markdown(
                        metric_card(label, f"{val:.2%}", sub),
                        unsafe_allow_html=True,
                    )
                st.markdown(
                    f"<p style='color:#94a3b8; font-size:0.85rem; margin-top:0.6rem;'>"
                    f"Champion model: <b style='color:#a5b4fc;'>{model_report.get('best_model', 'N/A')}</b>"
                    f"</p>",
                    unsafe_allow_html=True,
                )

            # ── Visualisations ──
            img_paths = [
                PROJECT_ROOT / "reports" / "model" / "confusion_matrix.png",
                PROJECT_ROOT / "reports" / "explainability" / "feature_importance.png",
            ]
            existing_imgs = [p for p in img_paths if p.exists()]
            if existing_imgs:
                st.markdown("---")
                st.markdown(
                    '<div class="section-header"><h3>🖼 Visualisations</h3></div>',
                    unsafe_allow_html=True,
                )
                vis_cols = st.columns(len(existing_imgs))
                for col, img in zip(vis_cols, existing_imgs):
                    col.image(str(img), caption=img.stem.replace("_", " ").title(), use_container_width=True)

            # ── Downloads ──
            st.markdown("---")
            st.markdown(
                '<div class="section-header"><h3>📦 Download Artifacts</h3></div>',
                unsafe_allow_html=True,
            )
            artifacts = {
                "🌐 Final HTML Report":     (PROJECT_ROOT / "reports" / "final_report.html", "text/html"),
                "🤖 Trained Model (.joblib)":(PROJECT_ROOT / "reports" / "model" / "best_model.joblib", "application/octet-stream"),
                "🧪 Test Script (.py)":     (PROJECT_ROOT / "reports" / "model" / "test_model.py", "text/x-python"),
                "📖 Integration Guide":     (PROJECT_ROOT / "reports" / "model" / "integration_guide.md", "text/markdown"),
                "📊 Feature Importance":    (PROJECT_ROOT / "reports" / "explainability" / "feature_importance.csv", "text/csv"),
                "🖼 Confusion Matrix":      (PROJECT_ROOT / "reports" / "model" / "confusion_matrix.png", "image/png"),
            }
            dl_cols = st.columns(3)
            for i, (label, (path, mime)) in enumerate(artifacts.items()):
                if path.exists():
                    dl_cols[i % 3].download_button(
                        label=label,
                        data=path.read_bytes(),
                        file_name=path.name,
                        mime=mime,
                        key=f"dl_{path.name}",
                        use_container_width=True,
                    )


# ──────────────────────────────────────────────
#  PAGE: PREDICTION SANDBOX
# ──────────────────────────────────────────────
elif page_name == "Prediction Sandbox":
    st.markdown(
        """
        <div class="hero-banner">
            <p class="hero-title" style="font-size:1.6rem;">🧪 Live Prediction Sandbox</p>
            <p class="hero-sub">Test the trained model instantly with any custom input.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_path = PROJECT_ROOT / "reports" / "model" / "best_model.joblib"
    test_info_path = PROJECT_ROOT / "reports" / "model" / "test_info.json"

    if not model_path.exists():
        st.info("🤖 No trained model found yet. Run **Analyse Dataset** first to train a model.")
        st.stop()

    test_info: dict = {}
    if test_info_path.exists():
        try:
            test_info = json.loads(test_info_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    task_title   = test_info.get("task_summary", "Trained Classification Model")
    default_msg  = test_info.get("sample_input", "Enter a sample message here...")
    custom_code  = test_info.get(
        "custom_code",
        "import joblib\nimport pandas as pd\n\nmodel = joblib.load('best_model.joblib')\ninput_df = pd.DataFrame({'__combined_text__': ['YOUR_TEXT']})\nprint(model.predict(input_df)[0])",
    )

    st.markdown(
        f"<p style='color:#94a3b8; margin-bottom:1.2rem;'>🎯 <b style='color:#a5b4fc;'>Task:</b> {task_title}</p>",
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.1, 0.9], gap="large")

    with left_col:
        st.markdown('<div class="card-title">📝 Enter Your Test Message</div>', unsafe_allow_html=True)
        test_input = st.text_area(
            "Test message",
            value=default_msg,
            height=130,
            label_visibility="collapsed",
            key="test_message_input",
            placeholder="Type or paste any message to classify…",
        )

        sample_cols = st.columns(2)
        spam_samples = [
            "WINNER! You've been selected for a FREE prize. Call now!",
            "Urgent: Your account will be closed. Click to verify immediately.",
            "Congratulations! You won £1000 cash prize. Text WIN to 83600.",
        ]
        ham_samples = [
            "Hey, are you coming to the meeting tomorrow?",
            "Can you pick up some groceries on your way home?",
            "I'll be late for dinner tonight, don't wait for me.",
        ]
        with sample_cols[0]:
            if st.button("🚫 Try Spam Sample", use_container_width=True, type="secondary"):
                st.session_state["test_message_input"] = spam_samples[
                    int(time.time()) % len(spam_samples)
                ]
                st.rerun()
        with sample_cols[1]:
            if st.button("✅ Try Ham Sample", use_container_width=True, type="secondary"):
                st.session_state["test_message_input"] = ham_samples[
                    int(time.time()) % len(ham_samples)
                ]
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)

    with right_col:
        st.markdown('<div class="card-title">📊 Prediction Result</div>', unsafe_allow_html=True)
        result_placeholder = st.empty()
        result_placeholder.markdown(
            '<div class="glass-card" style="text-align:center; color:#475569; padding:2.5rem;">'
            '  <div style="font-size:2.5rem;">🎯</div>'
            '  <div style="margin-top:0.5rem; font-size:0.9rem;">Prediction will appear here</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        if predict_btn:
            if not test_input.strip():
                st.warning("Please enter a message to classify.")
            else:
                with st.spinner("Running prediction…"):
                    try:
                        result = predict_message(test_input, str(model_path))
                        pred = result.get("prediction", "Unknown")
                        score_key = "confidence" if "confidence" in result else "decision_score"
                        score = result.get(score_key, "N/A")

                        is_spam = str(pred).lower() == "spam"
                        badge_class = "pred-spam" if is_spam else "pred-ham"
                        emoji = "🚫" if is_spam else "✅"

                        result_placeholder.markdown(
                            f'<div class="pred-box {badge_class}">'
                            f'  {emoji} {pred.upper()}'
                            f'</div>'
                            f'<div class="glass-card" style="margin-top:0.8rem;">'
                            f'  <div class="metric-label">Decision Score</div>'
                            f'  <div style="font-size:1.3rem; font-weight:700; color:#a5b4fc; margin-top:4px;">{score}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                        if "class_probabilities" in result:
                            probs = result["class_probabilities"]
                            st.markdown(f"**Class Probabilities:** `{probs}`")

                    except Exception as err:
                        result_placeholder.error(f"Prediction error: {err}")

    st.markdown("---")
    st.markdown('<div class="section-header"><h3>💻 Integration Code</h3></div>', unsafe_allow_html=True)
    st.caption("Copy this snippet to use the model in your own Python project:")
    st.code(custom_code, language="python")


# ──────────────────────────────────────────────
#  PAGE: MODEL REPORTS
# ──────────────────────────────────────────────
elif page_name == "Model Reports":
    st.markdown(
        """
        <div class="hero-banner">
            <p class="hero-title" style="font-size:1.6rem;">📊 Model Reports & Artifacts</p>
            <p class="hero-sub">Review all generated visualisations, metrics, and downloadable files.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model_report_path = PROJECT_ROOT / "reports" / "model" / "model_report.json"
    if not model_report_path.exists():
        st.info("No report found. Run **Analyse Dataset** first.")
        st.stop()

    report = json.loads(model_report_path.read_text(encoding="utf-8"))
    metrics = report.get("best_model_metrics", {})
    cls_report = metrics.get("classification_report", {})

    # ── Top metrics ──
    m1, m2, m3, m4 = st.columns(4)
    for col, label, key, sub in [
        (m1, "Accuracy",  "accuracy",          "Test set"),
        (m2, "Precision", "precision_weighted", "Weighted"),
        (m3, "Recall",    "recall_weighted",    "Weighted"),
        (m4, "F1 Score",  "f1_weighted",        "Weighted"),
    ]:
        col.markdown(
            metric_card(label, f"{metrics.get(key, 0):.2%}", sub),
            unsafe_allow_html=True,
        )

    st.markdown(
        f"<p style='color:#94a3b8; font-size:0.85rem; margin: 0.6rem 0 1.2rem;'>"
        f"Champion: <b style='color:#a5b4fc;'>{report.get('best_model','N/A')}</b> &nbsp;|&nbsp; "
        f"Training rows: <b style='color:#a5b4fc;'>{report.get('training_rows','N/A'):,}</b> &nbsp;|&nbsp; "
        f"Testing rows: <b style='color:#a5b4fc;'>{report.get('testing_rows','N/A'):,}</b>"
        f"</p>",
        unsafe_allow_html=True,
    )

    # ── Per-class table ──
    st.markdown('<div class="section-header"><h3>📋 Per-Class Metrics</h3></div>', unsafe_allow_html=True)
    table_rows = []
    for cls, vals in cls_report.items():
        if isinstance(vals, dict):
            table_rows.append({
                "Class": cls,
                "Precision": f"{vals.get('precision', 0):.4f}",
                "Recall":    f"{vals.get('recall', 0):.4f}",
                "F1-Score":  f"{vals.get('f1-score', 0):.4f}",
                "Support":   int(vals.get("support", 0)),
            })
    if table_rows:
        st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    # ── Images ──
    st.markdown("---")
    st.markdown('<div class="section-header"><h3>🖼 Visualisations</h3></div>', unsafe_allow_html=True)
    all_images = {
        "Confusion Matrix":    PROJECT_ROOT / "reports" / "model" / "confusion_matrix.png",
        "Feature Importance":  PROJECT_ROOT / "reports" / "explainability" / "feature_importance.png",
        "Numeric Distributions": PROJECT_ROOT / "reports" / "eda" / "numeric_distributions.png",
        "Category Distributions": PROJECT_ROOT / "reports" / "eda" / "categorical_distributions.png",
        "Correlation Heatmap": PROJECT_ROOT / "reports" / "eda" / "correlation_heatmap.png",
    }
    img_cols = st.columns(3)
    for i, (title, path) in enumerate(all_images.items()):
        if path.exists():
            img_cols[i % 3].image(str(path), caption=title, use_container_width=True)

    # ── Downloads ──
    st.markdown("---")
    st.markdown('<div class="section-header"><h3>📦 Download All Artifacts</h3></div>', unsafe_allow_html=True)
    artifacts = {
        "🌐 HTML Report":          (PROJECT_ROOT / "reports" / "final_report.html", "text/html"),
        "🤖 Trained Model":        (PROJECT_ROOT / "reports" / "model" / "best_model.joblib", "application/octet-stream"),
        "🧪 Test Script":          (PROJECT_ROOT / "reports" / "model" / "test_model.py", "text/x-python"),
        "📖 Integration Guide":    (PROJECT_ROOT / "reports" / "model" / "integration_guide.md", "text/markdown"),
        "📊 Feature Importance":   (PROJECT_ROOT / "reports" / "explainability" / "feature_importance.csv", "text/csv"),
        "📁 Model Report JSON":    (PROJECT_ROOT / "reports" / "model" / "model_report.json", "application/json"),
    }
    dl_cols = st.columns(3)
    for i, (label, (path, mime)) in enumerate(artifacts.items()):
        if path.exists():
            dl_cols[i % 3].download_button(
                label=label, data=path.read_bytes(),
                file_name=path.name, mime=mime,
                key=f"rpt_dl_{path.name}", use_container_width=True,
            )


# ──────────────────────────────────────────────
#  PAGE: ABOUT
# ──────────────────────────────────────────────
elif page_name == "About":
    st.markdown(
        """
        <div class="hero-banner">
            <p class="hero-title" style="font-size:1.6rem;">ℹ️ About This Project</p>
            <p class="hero-sub">Autonomous Data Science Analyst Agent — System Architecture</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown(
            """
            <div class="glass-card">
            <div class="card-title">🏗 Architecture — 7 Agent Pipeline</div>
            <div style="font-size:0.88rem; color:#cbd5e1; line-height:1.9;">
            <b style="color:#a5b4fc;">Agent 1</b> — Data Profiler<br>
            <span style="color:#64748b; font-size:0.8rem;">Schema, dtypes, missingness, cardinality</span><br>
            <b style="color:#a5b4fc;">Agent 2</b> — LLM Orchestrator<br>
            <span style="color:#64748b; font-size:0.8rem;">Pydantic-validated task plan via Qwen3/Gemini</span><br>
            <b style="color:#a5b4fc;">Agent 3</b> — Data Cleaning Agent<br>
            <span style="color:#64748b; font-size:0.8rem;">Deduplication, imputation, singleton pruning</span><br>
            <b style="color:#a5b4fc;">Agent 4</b> — EDA Engine<br>
            <span style="color:#64748b; font-size:0.8rem;">Auto-generates distribution plots & heatmaps</span><br>
            <b style="color:#a5b4fc;">Agent 5</b> — Model Trainer<br>
            <span style="color:#64748b; font-size:0.8rem;">TF-IDF + LinearSVC vs LogisticRegression</span><br>
            <b style="color:#a5b4fc;">Agent 6</b> — Explainability Agent<br>
            <span style="color:#64748b; font-size:0.8rem;">Linear coefficient → token attribution</span><br>
            <b style="color:#a5b4fc;">Agent 7</b> — Report Synthesiser<br>
            <span style="color:#64748b; font-size:0.8rem;">Self-contained HTML + test code generation</span>
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_b:
        st.markdown(
            """
            <div class="glass-card">
            <div class="card-title">⚡ Tech Stack</div>
            <div style="font-size:0.88rem; color:#cbd5e1; line-height:2.0;">
            🔗 <b>LangGraph 1.0</b> — Agent orchestration DAG<br>
            🤖 <b>LangChain Core 1.6</b> — LLM abstraction layer<br>
            🧠 <b>Qwen3-14B</b> — Local private LLM (via Bionic)<br>
            ☁️ <b>Gemini 2.5 Flash</b> — Cloud LLM fallback<br>
            🛡 <b>Pydantic v2</b> — Schema-validated structured output<br>
            🐼 <b>Pandas 2.x</b> — Data wrangling & profiling<br>
            📐 <b>Scikit-Learn 1.7</b> — ML pipelines & evaluation<br>
            📊 <b>Matplotlib + Seaborn</b> — Visualisation<br>
            🌐 <b>Streamlit 1.63</b> — This interactive dashboard
            </div>
            </div>

            <div class="glass-card">
            <div class="card-title">🔬 Key Research Contributions</div>
            <div style="font-size:0.84rem; color:#94a3b8; line-height:1.8;">
            ✦ Anti-hallucination via schema-grounded Pydantic plans<br>
            ✦ Privacy-preserving local LLM inference (zero data leakage)<br>
            ✦ Singleton-class noise detection preventing sklearn crashes<br>
            ✦ Integrated XAI through linear coefficient attribution<br>
            ✦ End-to-end < 3.2s on CPU — no GPU required
            </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
