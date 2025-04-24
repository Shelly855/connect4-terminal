"""
test_ml_prediction.py - Simple test to verify ML agent prediction.

Loads the basic ML agent model and prints a prediction for an empty board.
Used to confirm the model loads and returns a valid column index.
"""

import joblib

model = joblib.load("ml_agent.pkl")
test_input = [[0] * 42]
print("Example prediction:", model.predict(test_input)) # should output a number
