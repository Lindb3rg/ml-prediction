FROM python:3.9-slim
WORKDIR /app
COPY ml_prediction.py .
RUN pip install requests prometheus_client
EXPOSE 8000
CMD ["python", "ml_prediction.py"]