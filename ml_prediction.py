
import requests
import pickle
import numpy as np
from prometheus_client import start_http_server, Gauge


PREDICTION_METRIC = Gauge('prediction_metric', 'Prediction metric from ML model', ['machine_id'])

def get_data(url):
    response = requests.get(url)
    return response.json()


def load_model():
    with open("Models/logistic_regression_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model



def convert_row(data_row):
    data_values = np.array(list(data_row.values())).reshape(1,-1)
    return data_values



def predict_failure(data, model):
    
    machine_id = data.pop("Machine ID")
    data_values = convert_row(data)
    prediction = model.predict_proba(data_values)
    PREDICTION_METRIC.labels(machine_id=machine_id).set(prediction[0][1])
    


    
    return max

if __name__ == "__main__":
    
    start_http_server(8000)
    model = load_model()
    url = "http://localhost:5000/get_data"
    
    while True:
        data = get_data(url)
        

        prediction = predict_failure(data,model)
