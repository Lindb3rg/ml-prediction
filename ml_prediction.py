
import requests
import pickle
import numpy as np

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
    
    data = convert_row(data)
    prediction = model.predict_proba(data)
    max = prediction.max()
    print(max)

    
    return prediction

if __name__ == "__main__":
    
    model = load_model()
        
    url = "http://localhost:5000/get_data"
    
    while True:
        data = get_data(url)
        prediction = predict_failure(data,model)
