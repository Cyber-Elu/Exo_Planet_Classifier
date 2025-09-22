FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY templates ./templates
COPY models ./models
COPY data ./data
COPY scripts ./scripts

ENV MODEL_PATH=models/exoplanet_clf.joblib \
    DB_PATH=exoplanetes.db \
    API_HOST=0.0.0.0 \
    API_PORT=5000

EXPOSE 5000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "src.web:app"]
