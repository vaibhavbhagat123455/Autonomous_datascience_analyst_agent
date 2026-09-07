# Autonomous Data Science Analyst Agent

A real-data, LangGraph-orchestrated data science system powered by Google Gemini.

The system accepts a CSV or Excel dataset and a natural-language question. It then profiles, cleans, analyzes, models, explains, and reports the results.

## Architecture

```text
Real Dataset + Natural-Language Question
                  |
                  v
          Data Profiler Agent
                  |
                  v
          Gemini Orchestrator
                  |
                  v
           Cleaning Agent
                  |
                  v
              EDA Agent
                  |
                  v
        Model Selector/Trainer
                  |
                  v
        Explainability Agent
                  |
                  v
             Report Agent
                  |
                  v
          HTML Report + Model
```

## Technology stack

- Python
- LangGraph
- LangChain
- Google Gemini
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Joblib

## Project structure

```text
DSA/
├── data/
│   ├── email.csv
│   └── processed/
├── reports/
│   ├── eda/
│   ├── model/
│   ├── explainability/
│   └── final_report.html
├── src/
│   ├── data_loader.py
│   ├── state.py
│   ├── graph.py
│   ├── orchestrator.py
│   ├── cleaner.py
│   ├── eda.py
│   ├── model_trainer.py
│   ├── explainability.py
│   ├── report_agent.py
│   ├── predictor.py
│   └── predict_cli.py
├── tests/
├── .env
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

Create and activate the virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create `.env` from `.env.example` and add your Google Gemini API key:

```text
GOOGLE_API_KEY=your_actual_key
GEMINI_MODEL=gemini-2.5-flash
```

## Run the complete workflow

```powershell
python -m src.main `
  --file "data\email.csv" `
  --question "Build a classification model to predict Category from Message. Compare the models and report accuracy, precision, recall, and F1 score."
```

## Generated outputs

### Cleaned dataset

```text
data/processed/cleaned_dataset.csv
```

### EDA

```text
reports/eda/descriptive_statistics.csv
reports/eda/numeric_distributions.png
reports/eda/correlation_heatmap.png
reports/eda/categorical_distributions.png
reports/eda/eda_report.json
```

### Model

```text
reports/model/best_model.joblib
reports/model/model_comparison.json
reports/model/model_report.json
reports/model/confusion_matrix.png
```

### Explainability

```text
reports/explainability/feature_importance.csv
reports/explainability/feature_importance.png
reports/explainability/explainability_report.json
```

### Final report

```text
reports/final_report.html
```

## Predict a new message

Run the complete workflow first so that the model exists, then run:

```powershell
python -m src.predict_cli `
  --message "Congratulations! You won a free prize. Click now to claim it."
```

Example output:

```text
Prediction result
-----------------
Prediction: spam
Decision score: 0.8650
```

## Run tests

```powershell
pytest -q
```

## Important data-quality decision

The dataset contained one corrupted target label:

```text
{"mode":"full"
```

Because it appeared only once, it was excluded from stratified model training and recorded in `reports/model/model_report.json`. The valid `ham` and `spam` classes were retained.

## Limitations

- The current trainer implements classification for the demonstrated email dataset.
- The model is evaluated on a held-out test split and should be validated on new real-world data before deployment.
- The HTML report is intended for local project demonstration.
- API keys must remain in `.env` and must never be committed.
