# Trained Model Integration Guide

The trained model is `linear_svm` and is saved as `best_model.joblib`.

## Installation

```bash
pip install pandas joblib scikit-learn
```

## Load and use the model

```python
import joblib
import pandas as pd

model = joblib.load('best_model.joblib')
message = 'Your message text goes here'
input_data = pd.DataFrame({'__combined_text__': [message]})
prediction = model.predict(input_data)[0]
print(prediction)
```

## Important

This model expects the same text preprocessing used during training. For this project, pass the message as the `__combined_text__` feature. Use the model only for the same type of text classification task and label structure.
