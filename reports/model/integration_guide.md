# Trained Model Integration Guide (Spam Detection)

The trained model is `linear_svm` and is saved as `best_model.joblib`.

## Quick Standalone Testing (`test_model.py`)

A standalone tester script `test_model.py` is saved in the same directory as `best_model.joblib`.

To test custom input messages interactively in your terminal:

```bash
python test_model.py
```

## Installation

```bash
pip install pandas joblib scikit-learn
```

## Load and use the model programmatically

```python
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('best_model.joblib')

# Prepare input data using the required '__combined_text__' formatting
data = pd.DataFrame({
    '__combined_text__': [
        "Congratulations! You've been selected to win a free vacation to Hawaii. Click the link to claim your prize now!",
        "Hey, are we still meeting for lunch tomorrow at 12 PM? Let me know if you need to reschedule."
    ]
})

# Make predictions for target 'Category'
predictions = model.predict(data)

# Output the results
for text, pred in zip(data['__combined_text__'], predictions):
    print(f"Message: {text[:50]}... --> Prediction: {pred}")
```

## Sample Input Examples

- **Sample 1**: `Congratulations! You've been selected to win a free vacation to Hawaii. Click the link to claim your prize now!`
- **Sample 2**: `Hey, are we still meeting for lunch tomorrow at 12 PM? Let me know if you need to reschedule.`

## Important

This model expects text preprocessing via the `__combined_text__` feature column. Use the model for classification matching target column `Category`.
