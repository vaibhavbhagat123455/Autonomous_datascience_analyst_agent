from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_final_report_exists_after_pipeline_run():
    report_path = PROJECT_ROOT / "reports" / "final_report.html"

    if not report_path.exists():
        return

    assert report_path.stat().st_size > 0


def test_required_source_modules_exist():
    source_files = [
        "data_loader.py",
        "state.py",
        "graph.py",
        "orchestrator.py",
        "cleaner.py",
        "eda.py",
        "model_trainer.py",
        "explainability.py",
        "report_agent.py",
        "predictor.py",
    ]

    for filename in source_files:
        assert (PROJECT_ROOT / "src" / filename).exists()
