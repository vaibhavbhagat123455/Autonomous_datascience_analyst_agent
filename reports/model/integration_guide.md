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
import joblib
model = joblib.load('best_model.joblib')
text = "Sample email text here"
prediction = model.predict([[text]])
print(f'Predicted category: {prediction[0]}')
```

## Sample Input Examples

- **Sample 1**: `Meeting confirmation: Please find attached the agenda.`
- **Sample 2**: `URGENT! Your account will be closed unless you click here: http://maliciouslink.com`

## Important

This model expects text preprocessing via the `__combined_text__` feature column. Use the model for classification matching target column `Category`.
