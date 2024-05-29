from flask import Flask, jsonify, request
import pickle
import numpy as np
from prometheus_client import start_http_server, Gauge


app = Flask(__name__)

PREDICTION_METRIC = Gauge('prediction_metric', 'Prediction metric from ML model', ['machine_id'])


def load_model():
    with open("Models/logistic_regression_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model


def convert_row(data_row):
    data_values = np.array(list(data_row.values())).reshape(1,-1)
    return data_values


@app.route('/predict', methods=['POST'])
def predict():
    model = load_model()
    data = request.json.get('data')
    print(data)
    if data:
        print(data)
        machine_id = data.pop("Machine ID")
        data_values = convert_row(data)
        prediction = model.predict_proba(data_values)
        PREDICTION_METRIC.labels(machine_id=machine_id).set(prediction[0][1])
        
        print("******* PREDICTION *******", prediction[0][1])
        
        return jsonify({'prediction': prediction[0][1]}), 200
    
    return jsonify({'message': 'No data received'}), 400


if __name__ == "__main__":
    
    app.run(port=5001, debug=True)
    
    start_http_server(8000)

        
