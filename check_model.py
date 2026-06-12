# check_model.py

import pickle

print("Loading model...")

with open("loan_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

print("Model Loaded Successfully")