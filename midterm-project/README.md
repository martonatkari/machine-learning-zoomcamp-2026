# Student Depression Prediction

A binary classifier that predicts whether a student reports depression from survey answers (academic pressure,
sleep, diet, financial stress, ...), served as a Flask web service in a Docker container.

```
 survey answers (JSON)          scikit-learn pipeline               response (JSON)
┌────────────────────┐  POST   ┌────────────────────────────┐      ┌───────────────────────────┐
│ age, sleep, diet,  │ ──────► │ impute + scale + one-hot   │ ───► │ depression_probability,   │
│ pressure, ...      │ /predict│ → logistic regression      │      │ prediction, label         │
└────────────────────┘         └────────────────────────────┘      └───────────────────────────┘
        Flask (serve.py) → validation (predict.py) → artifacts/model.joblib   [Docker, port 9696]
```

## 1. Problem statement

A university counselling service cannot interview every student. A model that estimates depression risk from a short
questionnaire could help **prioritise outreach** or power a quick self-screening form. The model outputs a probability
and a label (probability ≥ 0.5).

**Metric.** The classes are mildly imbalanced (58.5% positive), so models are selected by **ROC AUC** (5-fold CV) and
additionally compared on accuracy, precision, recall and F1 on a held-out 20% test set. For screening, **recall** matters
most: missing an at-risk student is worse than a false alarm.

> **This is a course project, not a diagnostic tool.** See the limitations below.

## 2. Dataset

[Student Depression Dataset](https://www.kaggle.com/datasets/hopesb/student-depression-dataset) (Kaggle).
27,901 rows, 18 columns, target `Depression` (0/1). Kaggle tags it as *synthetic*; the respondents are Indian students.
A copy is included at `data/student_depression_dataset.csv` (2.7 MB), so the project runs offline.

Cleaning (`features.py → load_data`):

- keep only `Profession == "Student"` (31 rows are other professions) → **27,870 rows**,
- drop `id`, `Work Pressure`, `Job Satisfaction` (zero for almost every student),
- rename columns to `snake_case`,
- 3 missing `financial_stress` values are median-imputed inside the pipeline.

## 3. EDA summary

Full analysis with plots: [`notebook.ipynb`](student-depression/notebook.ipynb) (outputs are included, no need to re-run).

- No duplicates; almost no missing values (3 in `financial_stress`).
- Strongest numeric signals: `academic_pressure` (r = 0.47), `financial_stress` (0.36), `age` (−0.23),
  `work_study_hours` (0.21). `cgpa` is almost uncorrelated (0.02).
- Poor sleep (< 5 h: 65% depressed vs 51% for > 8 h) and unhealthy diet (71% vs 45% healthy) matter; gender does not.
- **Feature selection:** `city` contains junk values (names, `3.0`, ...), `degree` has 28 levels and `gender` shows no
  relation with the target. Cross-validated with the same logistic regression, the model with all 13 candidate features
  and the one without `city`, `degree` and `gender` score the same ROC AUC (0.9212 vs 0.9212, fold-to-fold std ≈ 0.003).
  These three columns are therefore **dropped**: 10 features remain, and the API does not need to handle junk or
  unseen cities. The remaining categorical features are one-hot encoded (rare / unseen values go into an "infrequent" bucket).
- **Feature-leakage check:** `suicidal_thoughts` is by far the strongest feature (79% vs 23% depression rate). It is a
  separate survey answer, not derived from the label, but it is tightly tied to depression. Every model is therefore
  trained on two feature sets, **`full`** and **`no_suicidal`**, and both are reported.

## 4. Modeling

Four models, each tuned with a grid search (5-fold stratified CV, scored by ROC AUC) on an 80% training split and
evaluated on the untouched 20% test split. Logistic regression is the baseline.

**Feature set `full` (deployed)**

| Model | CV ROC AUC | Test ROC AUC | Accuracy | Precision | Recall | F1 | Best params |
|---|---|---|---|---|---|---|---|
| **logreg (baseline)** | 0.921 | 0.922 | 0.846 | 0.858 | 0.883 | 0.870 | C=0.1 |
| decision_tree | 0.903 | 0.906 | 0.828 | 0.843 | 0.868 | 0.855 | max_depth=12, min_samples_leaf=100 |
| random_forest | 0.916 | 0.918 | 0.844 | 0.853 | 0.885 | 0.869 | max_depth=14, min_samples_leaf=5 |
| hist_gradient_boosting | 0.921 | 0.922 | 0.849 | 0.862 | 0.883 | 0.872 | l2_regularization=1.0, learning_rate=0.1, max_leaf_nodes=8 |

**Feature set `no_suicidal` (ablation)**

| Model | CV ROC AUC | Test ROC AUC | Accuracy | Precision | Recall | F1 | Best params |
|---|---|---|---|---|---|---|---|
| logreg (baseline) | 0.873 | 0.868 | 0.794 | 0.807 | 0.851 | 0.828 | C=0.03 |
| decision_tree | 0.852 | 0.850 | 0.785 | 0.791 | 0.859 | 0.824 | max_depth=12, min_samples_leaf=100 |
| random_forest | 0.869 | 0.866 | 0.797 | 0.805 | 0.861 | 0.832 | max_depth=20, min_samples_leaf=20 |
| hist_gradient_boosting | 0.874 | 0.869 | 0.797 | 0.810 | 0.852 | 0.831 | l2_regularization=0.0, learning_rate=0.1, max_leaf_nodes=8 |

**Selection.** Logistic regression and gradient boosting are practically tied (CV AUC 0.9211 vs 0.9207), the forest is
slightly and the single tree clearly behind. The **logistic regression is deployed**: same quality, much simpler and
faster, and its coefficients are interpretable (largest risk coefficients: suicidal thoughts, academic pressure,
financial stress, unhealthy diet, long study/work hours, short sleep). The selected configuration is refit on all
27,870 rows before saving. Removing `suicidal_thoughts` costs about 0.05 AUC; to deploy that variant instead run
`python train.py --feature-set no_suicidal` (the API then no longer requires that field).

## 5. Run it

### Locally

```bash
git clone https://github.com/martonatkari/machine-learning-zoomcamp-2026.git && cd midterm-project/student-depression
python3.13 -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python train.py            # ~2 min: compares models, writes artifacts/model.joblib + metrics.json (optional,
                           # the trained model is already committed)
python predict.py sample_request.json     # one prediction from the command line
python serve.py            # web service on http://localhost:9696
python test_service.py     # smoke tests (no running server needed)
```

To open the notebook: `pip install -r requirements-notebook.txt && jupyter notebook notebook.ipynb`.

### With Docker

```bash
docker build -t student-depression .
docker run -it --rm -p 9696:9696 student-depression
```

The image contains only the service (`features.py`, `predict.py`, `serve.py`, the model artifact); data and notebook are
excluded via `.dockerignore`. The `HEALTHCHECK` calls `GET /health`.

## 6. API

| Endpoint | Description |
|---|---|
| `GET /health` | returns `{"status": "ok"}` |
| `POST /predict` | JSON body with the 10 fields below → probability, prediction, label |

```bash
curl -X POST -H "Content-Type: application/json" \
     -d @sample_request.json \
     http://localhost:9696/predict
```

```json
{"depression_probability": 0.9872, "label": "depression risk", "prediction": 1}
```

Request fields (see `sample_request.json`):

| Field | Allowed values |
|---|---|
| `age` | number, 15–70 |
| `academic_pressure`, `study_satisfaction` | number, 0–5 |
| `cgpa` | number, 0–10 |
| `sleep_duration` | `Less than 5 hours`, `5-6 hours`, `7-8 hours`, `More than 8 hours`, `Others` |
| `dietary_habits` | `Healthy`, `Moderate`, `Unhealthy`, `Others` |
| `suicidal_thoughts`, `family_history` | `Yes`, `No` |
| `work_study_hours` | number, 0–24 |
| `financial_stress` | number, 1–5 (may be omitted/`null`, then it is imputed) |

Invalid input returns HTTP 422 with a message listing every problem; a non-JSON body returns 400. Additional fields
(e.g. `city`, `gender`, `degree`, which are not model features) are ignored.

## 7. Project structure

```
features.py            shared cleaning + feature definitions
train.py               model comparison, tuning, saves the pipeline
predict.py             loads the model, validates input, predicts
serve.py               Flask app (/health, /predict)
test_service.py        smoke tests for the API
notebook.ipynb         EDA, cleaning, model selection (with outputs)
artifacts/             model.joblib, metrics.json
data/                  dataset (CSV)
Dockerfile, requirements*.txt, sample_request.json
```

## 8. Limitations / next steps

- The data is **tagged as synthetic** and covers Indian students only; the performance does not transfer to real populations.
  The model must not be used to diagnose or make decisions about real people.
- Self-reported survey data; `Depression` is a label from the dataset, not a clinical assessment.
- The deployed model uses `suicidal_thoughts`, which makes it a *screening* helper that relies on a sensitive question. 
  The `no_suicidal` variant could be used if that is not acceptable.
- Some best parameters (tree depth, boosting learning rate and leaf size) lie at the edge of their grids; widening the
  tree / forest depth grids earlier changed the CV AUC by less than 0.001, so a larger search is unlikely to change the
  conclusion. A fuller search could still be run.
- Flask's built-in server is fine for this demo; for production use a WSGI server (e.g. gunicorn) behind a proxy.
- Next steps: probability calibration, a decision threshold tuned for recall, fairness checks across groups, cloud deployment.

## Reproducibility

The dependency versions in `requirements.txt` are pinned to the ones the committed `artifacts/model.joblib` was trained
with (a scikit-learn pickle must be loaded with the same version).
