from pathlib import Path

import pytest

from src.predictor import predict_message


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "reports" / "model" / "best_model.joblib"


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Run the full pipeline before testing the trained model.",
)
def test_trained_model_predicts_ham_message():
    result = predict_message(
        "Hey, are we still meeting tomorrow at 10 AM?",
        str(MODEL_PATH),
    )

    assert result["prediction"] in {"ham", "spam"}
    assert result["prediction"] == "ham"


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Run the full pipeline before testing the trained model.",
)
def test_trained_model_predicts_spam_message():
    result = predict_message(
        "Congratulations! You won a free prize. Click now to claim it.",
        str(MODEL_PATH),
    )

    assert result["prediction"] in {"ham", "spam"}
