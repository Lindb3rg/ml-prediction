FROM python:3.9-slim
WORKDIR /app
COPY ml_prediction.py .
COPY . .
RUN pip install flask requests prometheus_client numpy
EXPOSE 5001
EXPOSE 8000
CMD ["python", "ml_prediction.py"]