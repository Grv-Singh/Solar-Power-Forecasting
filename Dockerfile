FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt xgboost flask

COPY . .

EXPOSE 5000

ENV PORT=5000

CMD ["python3", "app.py"]
