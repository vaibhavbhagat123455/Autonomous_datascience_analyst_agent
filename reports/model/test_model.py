"""
Standalone Model Tester Script
Run this script to test custom inputs against the trained model in this folder.
"""
import json
from pathlib import Path
import joblib
import pandas as pd

MODEL_PATH = Path(__file__).parent / "best_model.joblib"
INFO_PATH = Path(__file__).parent / "test_info.json"

def run_test():
    if not MODEL_PATH.exists():
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    print(f"Loading trained model from {MODEL_PATH.name}...")
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!\n")

    test_info = {}
    if INFO_PATH.exists():
        try:
            test_info = json.loads(INFO_PATH.read_text(encoding="utf-8"))
            print(f"Task: {test_info.get('task_summary', 'Model Classification')}")
        except Exception:
            pass

    samples = []
    if "sample_input" in test_info:
        samples.append(test_info["sample_input"])
    if "sample_input_2" in test_info:
        samples.append(test_info["sample_input_2"])

    if samples:
        print("--- Preset Sample Predictions ---")
        for sample in samples:
            input_df = pd.DataFrame({"__combined_text__": [sample]})
            pred = model.predict(input_df)[0]
            score_str = ""
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)[0]
                score_str = f" (Confidence: {max(probs):.2%})"
            elif hasattr(model, "decision_function"):
                score = model.decision_function(input_df)[0]
                score_str = f" (Decision Score: {score:.4f})"
            print(f"Input: '{sample}'")
            print(f"--> Predicted Output: [{pred}]{score_str}\n")

    print("=" * 60)
    print("Enter a text message to test model prediction.")
    print("Type 'exit', 'quit', or press Ctrl+C to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            text = input("Enter message: ").strip()
            if text.lower() in ["exit", "quit", "q"]:
                print("Exiting model tester.")
                break
            if not text:
                continue

            input_df = pd.DataFrame({"__combined_text__": [text]})
            prediction = model.predict(input_df)[0]

            confidence = ""
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)[0]
                confidence = f" (Confidence: {max(probs):.2%})"
            elif hasattr(model, "decision_function"):
                score = model.decision_function(input_df)[0]
                confidence = f" (Decision Score: {score:.4f})"

            print(f"--> Prediction: [{prediction}]{confidence}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting model tester.")
            break

if __name__ == "__main__":
    run_test()
