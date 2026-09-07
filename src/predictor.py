from pathlib import Path

import joblib
import pandas as pd


def predict_message(
    message: str,
    model_path: str = "reports/model/best_model.joblib",
) -> dict[str, str]:
    """Predict whether a new message belongs to a trained target class."""

    if not message.strip():
        raise ValueError("Message cannot be empty.")

    model_file = Path(model_path)

    if not model_file.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_file}. Run the full pipeline first."
        )

    model_pipeline = joblib.load(model_file)

    # The training pipeline converts the Message column into this combined
    # text feature before applying TF-IDF.
    input_features = pd.DataFrame(
        {
            "__combined_text__": [message],
        }
    )

    prediction = str(model_pipeline.predict(input_features)[0])

    result = {
        "prediction": prediction,
        "message": message,
    }

    if hasattr(model_pipeline, "decision_function"):
        decision = model_pipeline.decision_function(input_features)
        if getattr(decision, "ndim", 1) == 1:
            result["decision_score"] = f"{float(decision[0]):.4f}"

    if hasattr(model_pipeline, "predict_proba"):
        probabilities = model_pipeline.predict_proba(input_features)[0]
        classes = model_pipeline.classes_

        result["confidence"] = f"{float(max(probabilities)):.4f}"
        result["class_probabilities"] = ", ".join(
            f"{class_name}={probability:.4f}"
            for class_name, probability in zip(classes, probabilities)
        )

    return result
