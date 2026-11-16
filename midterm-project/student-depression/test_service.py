#!/usr/bin/env python3
"""Smoke tests for the web service (uses Flask's test client, no server needed).

    python test_service.py
"""
import json
from pathlib import Path

from serve import app

SAMPLE = json.loads((Path(__file__).parent / "sample_request.json").read_text())
client = app.test_client()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.get_json() == {"status": "ok"}


def test_predict_ok():
    r = client.post("/predict", json=SAMPLE)
    body = r.get_json()
    assert r.status_code == 200
    assert 0.0 <= body["depression_probability"] <= 1.0
    assert body["prediction"] in (0, 1)


def test_risk_ordering():
    low = {**SAMPLE, "academic_pressure": 1, "financial_stress": 1,
           "sleep_duration": "7-8 hours", "dietary_habits": "Healthy", "suicidal_thoughts": "No"}
    p_high = client.post("/predict", json=SAMPLE).get_json()["depression_probability"]
    p_low = client.post("/predict", json=low).get_json()["depression_probability"]
    assert p_high > p_low


def test_extra_fields_are_ignored():
    # city / gender / degree are no longer model features; sending them must not break anything
    extra = {**SAMPLE, "city": "Szeged", "gender": "Female", "degree": "MBA"}
    r = client.post("/predict", json=extra)
    assert r.status_code == 200
    assert r.get_json() == client.post("/predict", json=SAMPLE).get_json()


def test_missing_financial_stress_is_imputed():
    body = {k: v for k, v in SAMPLE.items() if k != "financial_stress"}
    assert client.post("/predict", json=body).status_code == 200


def test_validation_errors():
    assert client.post("/predict", json={**SAMPLE, "age": 300}).status_code == 422
    assert client.post("/predict", json={**SAMPLE, "sleep_duration": "x"}).status_code == 422
    bad = {k: v for k, v in SAMPLE.items() if k != "cgpa"}
    assert client.post("/predict", json=bad).status_code == 422
    assert client.post("/predict", data="not json").status_code == 400


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} tests passed")
