# Trained Model Integration Guide (label Classifier)

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
import joblib
import pandas as pd

# Load trained model for target: label
model = joblib.load('best_model.joblib')

# Prepare text input DataFrame
sample_text = "Enter your test text sample here."
input_data = pd.DataFrame({'__combined_text__': [sample_text]})

# Predict output label (0, 1)
prediction = model.predict(input_data)[0]
print("Predicted label:", prediction)

```

## Sample Input Examples

- **Sample 1**: `Enter your test text sample here.`
- **Sample 2**: `Enter another sample input here.`

## Important

This model expects text preprocessing via the `__combined_text__` feature column. Use the model for classification matching target column `label`.
