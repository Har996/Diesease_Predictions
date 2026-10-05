FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train inside the image so the saved model always matches the scikit-learn version installed here.
# Uses data/disease_symptoms.csv (synthetic demo data unless you replaced it).
RUN python -m src.train --cv-folds 3

# Run as an unprivileged user; /app/instance holds the SQLite database.
RUN useradd --create-home appuser && mkdir -p /app/instance && chown -R appuser /app/instance
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["gunicorn", "app:create_app()", "--bind", "0.0.0.0:8000", "--workers", "2", "--access-logfile", "-"]
