FROM python:3.10

WORKDIR /app

# Copy requirements & wheels
COPY requirements.txt .
COPY wheels ./wheels

# Install packages OFFLINE
RUN pip install --no-index --find-links=./wheels -r requirements.txt

# Copy app code
COPY app ./app
COPY model.joblib .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
