#!/usr/bin/env python3
"""Flask web service exposing the model.

    python serve.py            # listens on 0.0.0.0:9696

Endpoints:
    GET  /health   -> {"status": "ok"}
    POST /predict  -> JSON in, prediction out (see README for an example)
"""
import os

from flask import Flask, jsonify, request

from predict import ValidationError, load_model, predict

app = Flask("student-depression")
load_model()  # fail fast at start-up if the model artifact is missing


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/predict")
def predict_endpoint():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify(error="Body must be valid JSON (Content-Type: application/json)."), 400
    try:
        return jsonify(predict(payload))
    except ValidationError as exc:
        return jsonify(error=str(exc)), 422


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 9696)))
