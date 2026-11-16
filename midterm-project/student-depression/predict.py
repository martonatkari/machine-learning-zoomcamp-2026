#!/usr/bin/env python3
"""Load the trained pipeline and run inference.

As a library:
    from predict import predict
    predict({"age": 21, "academic_pressure": 4, ...})  # see sample_request.json

From the command line:
    python predict.py sample_request.json
"""
import json
import sys
from functools import lru_cache

import joblib
import pandas as pd

import features as F

# Allowed values / ranges seen in the training data. Fields that are not model
# features (e.g. gender, city, degree) are simply ignored if a client sends them.
CHOICES = {
    "sleep_duration": {"Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours", "Others"},
    "dietary_habits": {"Healthy", "Moderate", "Unhealthy", "Others"},
    "suicidal_thoughts": {"Yes", "No"},
    "family_history": {"Yes", "No"},
}
RANGES = {
    "age": (15, 70),
    "academic_pressure": (0, 5),
    "cgpa": (0, 10),
    "study_satisfaction": (0, 5),
    "work_study_hours": (0, 24),
    "financial_stress": (1, 5),
}


class ValidationError(ValueError):
    """Raised for malformed input (the web service maps it to HTTP 422)."""


@lru_cache(maxsize=1)
def load_model():
    bundle = joblib.load(F.MODEL_PATH)
    return bundle["model"], bundle["features"]


def validate(record: dict, columns) -> dict:
    if not isinstance(record, dict):
        raise ValidationError("Request body must be a JSON object.")
    problems, clean = [], {}
    for col in columns:
        value = record.get(col)
        if value is None:
            if col == "financial_stress":  # the only column with missing values in training
                clean[col] = float("nan")
                continue
            problems.append(f"missing field: {col}")
            continue
        if col in RANGES:
            try:
                value = float(value)
            except (TypeError, ValueError):
                problems.append(f"{col} must be a number")
                continue
            lo, hi = RANGES[col]
            if not lo <= value <= hi:
                problems.append(f"{col} must be between {lo} and {hi}")
                continue
        elif col in CHOICES and value not in CHOICES[col]:
            problems.append(f"{col} must be one of {sorted(CHOICES[col])}")
            continue
        elif not isinstance(value, str):
            problems.append(f"{col} must be a string")
            continue
        clean[col] = value
    if problems:
        raise ValidationError("; ".join(problems))
    return clean


def predict(record: dict) -> dict:
    model, columns = load_model()
    row = pd.DataFrame([validate(record, columns)], columns=columns)
    proba = float(model.predict_proba(row)[0, 1])
    return {
        "depression_probability": round(proba, 4),
        "prediction": int(proba >= 0.5),
        "label": "depression risk" if proba >= 0.5 else "no depression risk",
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python predict.py <request.json>")
    with open(sys.argv[1]) as fh:
        print(json.dumps(predict(json.load(fh)), indent=2))
