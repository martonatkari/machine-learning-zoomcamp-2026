"""Shared data loading, cleaning and feature definitions.

Used by train.py, predict.py, serve.py and the notebook so that the exact same
preprocessing is applied during training and inference.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "student_depression_dataset.csv"
MODEL_PATH = ROOT / "artifacts" / "model.joblib"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"

TARGET = "depression"

RENAME = {
    "Gender": "gender",
    "Age": "age",
    "City": "city",
    "Profession": "profession",
    "Academic Pressure": "academic_pressure",
    "CGPA": "cgpa",
    "Study Satisfaction": "study_satisfaction",
    "Sleep Duration": "sleep_duration",
    "Dietary Habits": "dietary_habits",
    "Degree": "degree",
    "Have you ever had suicidal thoughts ?": "suicidal_thoughts",
    "Work/Study Hours": "work_study_hours",
    "Financial Stress": "financial_stress",
    "Family History of Mental Illness": "family_history",
    "Depression": TARGET,
}

# Columns dropped during cleaning:
#   id                 - row identifier, no signal
#   Work Pressure,
#   Job Satisfaction   - practically constant 0 for students (3 / 8 non-zero rows)
#   Profession         - only the 31 non-student rows differ; we keep students only
DROP_COLUMNS = ["id", "Work Pressure", "Job Satisfaction"]

NUMERIC = [
    "age",
    "academic_pressure",
    "cgpa",
    "study_satisfaction",
    "work_study_hours",
    "financial_stress",
]
CATEGORICAL = [
    "sleep_duration",
    "dietary_habits",
    "family_history",
]
SUICIDAL = "suicidal_thoughts"  # binary Yes/No, strongly related to the target

# Candidate features that were tested and removed: the notebook shows that
# dropping them leaves the cross-validated ROC AUC unchanged (0.9212 with and
# without them), so the simpler model (and API) is preferred. `city` also
# contains junk values.
EXCLUDED = ["gender", "city", "degree"]

FEATURES_FULL = NUMERIC + CATEGORICAL + [SUICIDAL]
FEATURES_NO_SUICIDAL = NUMERIC + CATEGORICAL
FEATURES_ALL_CANDIDATES = FEATURES_FULL + EXCLUDED  # only used for the ablation


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Load the CSV and apply the (deterministic) cleaning steps."""
    df = pd.read_csv(path)
    df = df[df["Profession"] == "Student"].copy()
    df = df.drop(columns=DROP_COLUMNS)
    df = df.rename(columns=RENAME).drop(columns=["profession"])
    return df.reset_index(drop=True)
