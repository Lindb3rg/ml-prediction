FROM python:3.9-slim
WORKDIR /app
COPY ml_prediction.py .
COPY requirements.txt .
COPY Models/logistic_regression_model.pkl /app/Models/
RUN pip install -r requirements.txt
EXPOSE 5001
EXPOSE 8000
CMD ["python", "ml_prediction.py"]