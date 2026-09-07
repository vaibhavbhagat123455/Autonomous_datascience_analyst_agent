import argparse

from .predictor import predict_message


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict the class of a new message."
    )
    parser.add_argument(
        "--message",
        required=True,
        help="The message to classify.",
    )
    parser.add_argument(
        "--model",
        default="reports/model/best_model.joblib",
        help="Path to the trained model.",
    )

    args = parser.parse_args()
    result = predict_message(args.message, args.model)

    print("\nPrediction result")
    print("-----------------")
    print(f"Prediction: {result['prediction']}")

    if "confidence" in result:
        print(f"Confidence: {result['confidence']}")

    if "class_probabilities" in result:
        print(f"Class probabilities: {result['class_probabilities']}")

    if "decision_score" in result:
        print(f"Decision score: {result['decision_score']}")


if __name__ == "__main__":
    main()
