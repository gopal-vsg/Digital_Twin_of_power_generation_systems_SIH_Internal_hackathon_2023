from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import random
import csv

app = Flask(__name__)

gas_model = joblib.load('gas_model.joblib')
steam_model = joblib.load('steam_model.joblib')



# Anomaly threshold (adjust as needed)
ANOMALY_THRESHOLD = 150

def read_dataset():
    with open('Anamoly.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        # Skip the header row if present
        next(csv_reader)
        for row in csv_reader:
            yield list(map(float, row))

dataset_generator = read_dataset()

@app.route('/')
def index():
    return render_template('anomaly.html')

@app.route('/predict', methods=['POST'])
def predict():
    global dataset_generator
    try:
        # Get the next row from the dataset
        try:
            row = next(dataset_generator)
        except StopIteration:
            # Restart the generator if the end of the dataset is reached
            dataset_generator = read_dataset()
            row = next(dataset_generator)
   
        

        CT_gas,V_gas,AP_gas,RH_gas,RPM_GAS_gas,EX_TEMP_gas,RPM_STEAM,STEAM_PRESSURE,WATER_FR,POWER_1,POWER_2= row

        user_input_gas = [[CT_gas, V_gas, AP_gas, RH_gas, RPM_GAS_gas, EX_TEMP_gas]]
        user_input_steam = [[EX_TEMP_gas, RPM_STEAM, STEAM_PRESSURE, WATER_FR]]

        predicted_ep_gas = gas_model.predict(user_input_gas)[0]
        predicted_ep_steam = steam_model.predict(user_input_steam)[0]

        predicted_ep_gas = predicted_ep_gas.tolist()
        predicted_ep_steam = predicted_ep_steam.tolist()

        anomaly_gas = bool(abs(predicted_ep_gas[0] - POWER_1) > ANOMALY_THRESHOLD)
        anomaly_steam = bool(abs(predicted_ep_steam[0] - POWER_2) > ANOMALY_THRESHOLD)

        response_data = {
            'CT_gas': CT_gas,
            'V_gas': V_gas,
            'AP_gas': AP_gas,
            'RH_gas': RH_gas,
            'RPM_GAS_gas': RPM_GAS_gas,
            'EX_TEMP_gas': EX_TEMP_gas,
            'predicted_ep_gas': float(predicted_ep_gas[0]),
            'RPM_STEAM': RPM_STEAM,
            'STEAM_PRESSURE': STEAM_PRESSURE,
            'WATER_FR': WATER_FR,
            'POWER_1': POWER_1,
            'POWER_2': POWER_2,
            'predicted_ep_steam': float(predicted_ep_steam[0]),
            'anomaly_gas': anomaly_gas,
            'anomaly_steam': anomaly_steam
        }
        

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port="5004")