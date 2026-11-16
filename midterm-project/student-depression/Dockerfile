FROM python:3.13-slim

WORKDIR /app

# install dependencies first so this layer is cached between code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# only what the web service needs (no data, no notebook)
COPY features.py predict.py serve.py ./
COPY artifacts/model.joblib artifacts/model.joblib

EXPOSE 9696

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9696/health')" || exit 1

CMD ["python", "serve.py"]
