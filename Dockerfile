FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

WORKDIR /app

COPY requirements-api.txt .

RUN python -m pip install \
    --no-cache-dir \
    -r requirements-api.txt

COPY src ./src
COPY models/hgb_churn_model_v1.joblib ./models/hgb_churn_model_v1.joblib

EXPOSE 8000

CMD ["sh", "-c", "python -m uvicorn src.api.main:app --host 0.0.0.0 --port $PORT"]