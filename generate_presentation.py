import collections.abc
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# Color Palette: Modern Professional Tech
NAVY = RGBColor(18, 54, 107)        # Primary deep blue #12366b
SLATE = RGBColor(30, 41, 59)        # Dark body text #1e293b
TEAL = RGBColor(14, 165, 233)       # Accent vibrant cyan/teal #0ea5e9
LIGHT_BG = RGBColor(248, 250, 252)  # Card background #f8fafc
CARD_BORDER = RGBColor(226, 232, 240)# Border grey #e2e8f0
WHITE = RGBColor(255, 255, 255)
GREEN = RGBColor(16, 185, 129)      # Success green #10b981
PURPLE = RGBColor(99, 102, 241)     # Indigo/Purple accent #6366f1

blank_layout = prs.slide_layouts[6]

def add_header(slide, title_text, category_text="AUTONOMOUS DATA SCIENCE ANALYST"):
    cat_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.7), Inches(0.4))
    tf_cat = cat_box.text_frame
    tf_cat.word_wrap = True
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = category_text.upper()
    p_cat.font.size = Pt(11)
    p_cat.font.bold = True
    p_cat.font.color.rgb = TEAL

    title_box = slide.shapes.add_textbox(Inches(0.8), Inches(0.7), Inches(11.7), Inches(0.8))
    tf_title = title_box.text_frame
    tf_title.word_wrap = True
    p_title = tf_title.paragraphs[0]
    p_title.text = title_text
    p_title.font.size = Pt(24)
    p_title.font.bold = True
    p_title.font.color.rgb = NAVY

def add_card(slide, left, top, width, height, bg_color=LIGHT_BG, border_color=CARD_BORDER):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = bg_color
    shape.line.color.rgb = border_color
    shape.line.width = Pt(1.5)
    return shape

# ==========================================
# SLIDE 1: Title Slide (Dark Premium Theme)
# ==========================================
slide1 = prs.slides.add_slide(blank_layout)
bg1 = slide1.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
bg1.fill.solid()
bg1.fill.fore_color.rgb = NAVY
bg1.line.fill.background()

tbox = slide1.shapes.add_textbox(Inches(1.0), Inches(1.8), Inches(11.333), Inches(3.5))
tf = tbox.text_frame
tf.word_wrap = True

p_badge = tf.paragraphs[0]
p_badge.text = "LANGGRAPH ORCHESTRATED MULTI-AGENT SYSTEM"
p_badge.font.size = Pt(14)
p_badge.font.bold = True
p_badge.font.color.rgb = TEAL
p_badge.space_after = Pt(14)

p_main = tf.add_paragraph()
p_main.text = "Autonomous Data Science Analyst Agent"
p_main.font.size = Pt(40)
p_main.font.bold = True
p_main.font.color.rgb = WHITE
p_main.space_after = Pt(18)

p_sub = tf.add_paragraph()
p_sub.text = "End-to-End Automated Data Profiling, Cleaning, EDA, Competitive ML Training, Explainability & Reporting"
p_sub.font.size = Pt(18)
p_sub.font.color.rgb = RGBColor(203, 213, 225)
p_sub.space_after = Pt(28)

p_foot = tf.add_paragraph()
p_foot.text = "Technology Stack: Python | LangGraph | Local LLMs (Qwen) / Gemini | Scikit-Learn | Streamlit"
p_foot.font.size = Pt(13)
p_foot.font.color.rgb = TEAL

# ==========================================
# SLIDE 2: Problem Statement & Motivation
# ==========================================
slide2 = prs.slides.add_slide(blank_layout)
add_header(slide2, "The Problem & Why Multi-Agent Systems Matter", "PROJECT MOTIVATION")

cards_data = [
    ("Traditional Bottleneck", "Over 70% of a Data Scientist's time is consumed by routine, manual tasks: parsing schemas, null imputation, formatting columns, and writing boilerplate evaluation code.", PURPLE),
    ("Limitations of Pure LLM Agents", "Pure LLMs instructed to 'write & run code' often hallucinate column names, fail on syntax errors, use unstratified splits, and crash on unexpected null types.", NAVY),
    ("Our Multi-Agent Solution", "LangGraph orchestrates a deterministic DAG (State Machine) where specialized agents handle dedicated sub-tasks reliably: Profiling, Cleaning, EDA, Model Training, Explainability, and Reporting.", GREEN)
]

for i, (title, desc, color) in enumerate(cards_data):
    left = Inches(0.8 + i * 4.0)
    add_card(slide2, left, Inches(1.7), Inches(3.7), Inches(5.0))

    tb = slide2.shapes.add_textbox(left + Inches(0.25), Inches(1.9), Inches(3.2), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = f"0{i+1}"
    p0.font.size = Pt(22)
    p0.font.bold = True
    p0.font.color.rgb = color
    p0.space_after = Pt(10)

    p1 = tf.add_paragraph()
    p1.text = title
    p1.font.size = Pt(18)
    p1.font.bold = True
    p1.font.color.rgb = NAVY
    p1.space_after = Pt(14)

    p2 = tf.add_paragraph()
    p2.text = desc
    p2.font.size = Pt(14)
    p2.font.color.rgb = SLATE

# ==========================================
# SLIDE 3: System Architecture (The 7-Agent Flow)
# ==========================================
slide3 = prs.slides.add_slide(blank_layout)
add_header(slide3, "System Architecture & End-to-End Pipeline", "GRAPH WORKFLOW")

flow_steps = [
    ("1. Data Profiler", "Inspects schema, row/col count, dtypes & nulls"),
    ("2. LLM Orchestrator", "Extracts target column, task type & requested metrics"),
    ("3. Cleaning Agent", "Drops duplicates, handles nulls & sanitizes labels"),
    ("4. EDA Agent", "Builds class balance & feature distribution charts"),
    ("5. Model Trainer", "Stratified TF-IDF split + LinearSVC vs Logistic Regression"),
    ("6. Explainability", "Extracts feature weights & top predictive token drivers"),
    ("7. Report Agent", "Generates self-contained Base64 HTML report + tester")
]

for i, (title, desc) in enumerate(flow_steps):
    y = Inches(1.6 + i * 0.76)
    add_card(slide3, Inches(0.8), y, Inches(11.7), Inches(0.66))

    tb = slide3.shapes.add_textbox(Inches(1.0), y + Inches(0.08), Inches(11.3), Inches(0.5))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = f"STAGE {i+1}: {title.upper()}"
    p0.font.size = Pt(13)
    p0.font.bold = True
    p0.font.color.rgb = NAVY

    p1 = tf.add_paragraph()
    p1.text = desc
    p1.font.size = Pt(11)
    p1.font.color.rgb = SLATE

# ==========================================
# SLIDE 4: Deterministic State Machine (AnalystState)
# ==========================================
slide4 = prs.slides.add_slide(blank_layout)
add_header(slide4, "State Machine & Shared Memory (AnalystState)", "STATE MANAGEMENT")

add_card(slide4, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.0))
tb_code = slide4.shapes.add_textbox(Inches(1.0), Inches(1.9), Inches(5.2), Inches(4.6))
tf_code = tb_code.text_frame
tf_code.word_wrap = True

p_c0 = tf_code.paragraphs[0]
p_c0.text = "AnalystState (TypedDict Schema)"
p_c0.font.size = Pt(16)
p_c0.font.bold = True
p_c0.font.color.rgb = NAVY
p_c0.space_after = Pt(10)

code_snippet = """class AnalystState(TypedDict, total=False):
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
    final_report_path: str
    status: str
    errors: list[str]"""

p_c1 = tf_code.add_paragraph()
p_c1.text = code_snippet
p_c1.font.size = Pt(11)
p_c1.font.color.rgb = SLATE

add_card(slide4, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.0))
tb_exp = slide4.shapes.add_textbox(Inches(7.0), Inches(1.9), Inches(5.3), Inches(4.6))
tf_exp = tb_exp.text_frame
tf_exp.word_wrap = True

p_e0 = tf_exp.paragraphs[0]
p_e0.text = "Architectural Advantages"
p_e0.font.size = Pt(18)
p_e0.font.bold = True
p_e0.font.color.rgb = NAVY
p_e0.space_after = Pt(14)

points = [
    ("Zero Information Loss", "Each agent writes its results into the state; downstream nodes access previous findings instantly."),
    ("Strict Determinism", "Typed contracts prevent schema mismatches and runtime serialization errors."),
    ("Safe Conditional Branching", "Edges like 'route_after_profiler' check if the dataset has valid columns. If invalid, the graph gracefully stops at END."),
    ("Auditability & Checkpointing", "Every transformation from raw rows to final metrics is tracked and recorded for compliance.")
]

for title, text in points:
    p_t = tf_exp.add_paragraph()
    p_t.text = f"• {title}:"
    p_t.font.size = Pt(13)
    p_t.font.bold = True
    p_t.font.color.rgb = PURPLE
    p_d = tf_exp.add_paragraph()
    p_d.text = f"   {text}"
    p_d.font.size = Pt(12)
    p_d.font.color.rgb = SLATE
    p_d.space_after = Pt(8)

# ==========================================
# SLIDE 5: Machine Learning & Feature Engineering
# ==========================================
slide5 = prs.slides.add_slide(blank_layout)
add_header(slide5, "Machine Learning Engine & Pipeline Design", "MODELING & EVALUATION")

ml_cards = [
    ("TF-IDF Vectorization", "Transforms unstructured text into mathematical n-gram vectors. Configured with (1, 2) unigrams and bigrams, sublinear TF scaling, and a 20,000 max feature cap to capture contextual word pairs without exploding dimensions."),
    ("Competitive Model Selection", "Trains both Logistic Regression and Linear Support Vector Classifier (LinearSVC) using stratified 80/20 train-test splits and balanced class weights to handle severe class imbalance (e.g. 88% ham vs 12% spam)."),
    ("Objective Evaluation Metrics", "Evaluates candidate models strictly on unseen held-out test data. Selects the champion model based on weighted F1-score, Precision, Recall, and Overall Accuracy, ensuring zero data leakage.")
]

for i, (title, desc) in enumerate(ml_cards):
    left = Inches(0.8 + i * 4.0)
    add_card(slide5, left, Inches(1.7), Inches(3.7), Inches(5.0))

    tb = slide5.shapes.add_textbox(left + Inches(0.25), Inches(1.9), Inches(3.2), Inches(4.5))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = title
    p0.font.size = Pt(17)
    p0.font.bold = True
    p0.font.color.rgb = NAVY
    p0.space_after = Pt(14)

    p1 = tf.add_paragraph()
    p1.text = desc
    p1.font.size = Pt(13)
    p1.font.color.rgb = SLATE

# ==========================================
# SLIDE 6: Explainability & Model Interpretability
# ==========================================
slide6 = prs.slides.add_slide(blank_layout)
add_header(slide6, "Explainable AI (XAI) & Interpretability", "FEATURE IMPORTANCE")

add_card(slide6, Inches(0.8), Inches(1.7), Inches(11.7), Inches(5.0))
tb_xai = slide6.shapes.add_textbox(Inches(1.1), Inches(1.9), Inches(11.1), Inches(4.6))
tf_xai = tb_xai.text_frame
tf_xai.word_wrap = True

p_x0 = tf_xai.paragraphs[0]
p_x0.text = "Opening the Black Box: How the Model Makes Decisions"
p_x0.font.size = Pt(19)
p_x0.font.bold = True
p_x0.font.color.rgb = NAVY
p_x0.space_after = Pt(14)

xai_bullets = [
    ("Mathematical Coefficient Extraction", "For linear models, coefficients directly reflect feature impact. Positive coefficients increase log-odds of a positive prediction, while negative coefficients suppress it."),
    ("Vocabulary Inverse Mapping", "Feature indices are mapped back to actual tokens (e.g., 'claim', 'free', 'urgent', 'winner') to extract the top positive and negative vocabulary drivers."),
    ("Feature Importance Visualization", "Exports 'feature_importance.png' and 'feature_importance.csv', providing transparency for stakeholders, end-users, and compliance auditors."),
    ("Local & Global Trust", "Ensures models rely on genuine semantic signals rather than training noise or confounding variables.")
]

for title, desc in xai_bullets:
    p_t = tf_xai.add_paragraph()
    p_t.text = f"✔ {title}"
    p_t.font.size = Pt(14)
    p_t.font.bold = True
    p_t.font.color.rgb = TEAL
    p_d = tf_xai.add_paragraph()
    p_d.text = f"    {desc}"
    p_d.font.size = Pt(12)
    p_d.font.color.rgb = SLATE
    p_d.space_after = Pt(8)

# ==========================================
# SLIDE 7: Dual LLM Architecture & Privacy
# ==========================================
slide7 = prs.slides.add_slide(blank_layout)
add_header(slide7, "Dual LLM Support: Local Privacy vs. Cloud Scale", "LLM INTEGRATION")

add_card(slide7, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.0))
tb_loc = slide7.shapes.add_textbox(Inches(1.1), Inches(1.9), Inches(5.0), Inches(4.6))
tf_loc = tb_loc.text_frame
tf_loc.word_wrap = True

p_l0 = tf_loc.paragraphs[0]
p_l0.text = "Option A: Local LLM (Bionic / LM Studio)"
p_l0.font.size = Pt(18)
p_l0.font.bold = True
p_l0.font.color.rgb = NAVY
p_l0.space_after = Pt(12)

local_points = [
    "Endpoint: http://localhost:1235/v1",
    "Model: Qwen3-14B / Qwen3-4B",
    "100% Data Privacy: No proprietary dataset records leave the local machine.",
    "Zero API Cost: Unlimited execution without external cloud token billing.",
    "Integrated via LangChain's ChatOpenAI adapter with timeout handling."
]
for pt in local_points:
    p = tf_loc.add_paragraph()
    p.text = f"• {pt}"
    p.font.size = Pt(12)
    p.font.color.rgb = SLATE
    p.space_after = Pt(6)

add_card(slide7, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.0))
tb_cld = slide7.shapes.add_textbox(Inches(7.1), Inches(1.9), Inches(5.1), Inches(4.6))
tf_cld = tb_cld.text_frame
tf_cld.word_wrap = True

p_c0 = tf_cld.paragraphs[0]
p_c0.text = "Option B: Cloud Scale (Google Gemini)"
p_c0.font.size = Pt(18)
p_c0.font.bold = True
p_c0.font.color.rgb = TEAL
p_c0.space_after = Pt(12)

cloud_points = [
    "Provider: Google Gemini (gemini-2.5-flash / gemini-1.5-pro)",
    "Cloud Scale & High Speed: Sub-second orchestration response times.",
    "Zero Local Hardware Load: Runs on thin clients without local GPU requirements.",
    "Strict JSON Schema Output: Leverages structured tool calling to guarantee valid Pydantic task plans.",
    "Seamless .env Toggle: Switch between local and cloud with a single flag."
]
for pt in cloud_points:
    p = tf_cld.add_paragraph()
    p.text = f"• {pt}"
    p.font.size = Pt(12)
    p.font.color.rgb = SLATE
    p.space_after = Pt(6)

# ==========================================
# SLIDE 8: Empirical Results & Verified Performance
# ==========================================
slide8 = prs.slides.add_slide(blank_layout)
add_header(slide8, "Empirical Validation & Benchmark Results", "EXPERIMENTAL PERFORMANCE")

metrics = [
    ("97.67%", "Overall Accuracy", "High true positive and true negative rate on held-out test data."),
    ("97.61%", "Weighted F1-Score", "Demonstrates strong precision-recall balance across imbalanced labels."),
    ("95.61%", "Spam Precision", "Minimal false positives ensure legitimate messages are never flagged."),
    ("< 3.0 sec", "Training Time", "Optimized sparse linear math converges in seconds on standard CPU.")
]

for i, (val, label, sub) in enumerate(metrics):
    left = Inches(0.8 + i * 3.0)
    add_card(slide8, left, Inches(1.7), Inches(2.7), Inches(2.5))

    tb = slide8.shapes.add_textbox(left + Inches(0.15), Inches(1.85), Inches(2.4), Inches(2.2))
    tf = tb.text_frame
    tf.word_wrap = True

    p0 = tf.paragraphs[0]
    p0.text = val
    p0.font.size = Pt(28)
    p0.font.bold = True
    p0.font.color.rgb = GREEN
    p0.space_after = Pt(4)

    p1 = tf.add_paragraph()
    p1.text = label
    p1.font.size = Pt(14)
    p1.font.bold = True
    p1.font.color.rgb = NAVY
    p1.space_after = Pt(4)

    p2 = tf.add_paragraph()
    p2.text = sub
    p2.font.size = Pt(10)
    p2.font.color.rgb = SLATE

# Bottom summary card
add_card(slide8, Inches(0.8), Inches(4.5), Inches(11.7), Inches(2.2))
tb_bot = slide8.shapes.add_textbox(Inches(1.1), Inches(4.65), Inches(11.1), Inches(1.9))
tf_bot = tb_bot.text_frame
tf_bot.word_wrap = True

p_b0 = tf_bot.paragraphs[0]
p_b0.text = "Production Artifact Deliverables"
p_b0.font.size = Pt(16)
p_b0.font.bold = True
p_b0.font.color.rgb = NAVY
p_b0.space_after = Pt(6)

artifacts_summary = [
    "best_model.joblib: Fully serialized Scikit-Learn Pipeline ready for microservice deployment.",
    "final_report.html: Self-contained executive report with Base64 embedded charts (no broken links).",
    "test_model.py: Standalone CLI testing script preloaded with dataset-specific examples.",
    "Interactive UI Sandbox: Live browser-based inference playground in Streamlit."
]
for art in artifacts_summary:
    p = tf_bot.add_paragraph()
    p.text = f"✔ {art}"
    p.font.size = Pt(11)
    p.font.color.rgb = SLATE

# ==========================================
# SLIDE 9: Conclusion & Future Scope
# ==========================================
slide9 = prs.slides.add_slide(blank_layout)
add_header(slide9, "Conclusion & Future Enhancements", "SUMMARY & OUTLOOK")

add_card(slide9, Inches(0.8), Inches(1.7), Inches(5.6), Inches(5.0))
tb_con = slide9.shapes.add_textbox(Inches(1.1), Inches(1.9), Inches(5.0), Inches(4.6))
tf_con = tb_con.text_frame
tf_con.word_wrap = True

p_c0 = tf_con.paragraphs[0]
p_c0.text = "Key Project Achievements"
p_c0.font.size = Pt(18)
p_c0.font.bold = True
p_c0.font.color.rgb = NAVY
p_c0.space_after = Pt(14)

achievements = [
    "Automated end-to-end data science lifecycle from raw upload to deployment.",
    "Guaranteed reliability through LangGraph deterministic DAG orchestration.",
    "Hybrid LLM design supporting both cloud scale and 100% offline local privacy.",
    "Robust data cleaning preventing runtime pipeline crashes on real-world noise.",
    "Fully interpretable AI output with coefficient importance and automated documentation."
]
for a in achievements:
    p = tf_con.add_paragraph()
    p.text = f"✔ {a}"
    p.font.size = Pt(12)
    p.font.color.rgb = SLATE
    p.space_after = Pt(8)

add_card(slide9, Inches(6.8), Inches(1.7), Inches(5.7), Inches(5.0))
tb_fut = slide9.shapes.add_textbox(Inches(7.1), Inches(1.9), Inches(5.1), Inches(4.6))
tf_fut = tb_fut.text_frame
tf_fut.word_wrap = True

p_f0 = tf_fut.paragraphs[0]
p_f0.text = "Future Roadmap & Extensions"
p_f0.font.size = Pt(18)
p_f0.font.bold = True
p_f0.font.color.rgb = PURPLE
p_f0.space_after = Pt(14)

future = [
    "Support for Tabular Regression & Multi-Class Classification pipelines.",
    "AutoML Integration: Hyperparameter search via Bayesian Optimization (Optuna).",
    "Transformer Embeddings: Fallback to local HuggingFace sentence transformers for nuanced tasks.",
    "One-Click REST API Export: Automatically wrap best_model.joblib in a containerized FastAPI endpoint."
]
for f in future:
    p = tf_fut.add_paragraph()
    p.text = f"➔ {f}"
    p.font.size = Pt(12)
    p.font.color.rgb = SLATE
    p.space_after = Pt(8)

output_pptx = "Autonomous_Data_Science_Analyst_Presentation.pptx"
prs.save(output_pptx)
print(f"Presentation saved successfully as '{output_pptx}'")
