from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import random
import csv
import numpy as np

app = Flask(__name__)

gas_model = joblib.load('gas_model.joblib')
steam_model = joblib.load('steam_model.joblib')

data = pd.read_csv("datasets\Folds5x2_pp.csv")
ANOMALY_THRESHOLD = 150

def read_dataset():
    with open('datasets\Anamoly.csv', 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        # Skip the header row if present
        next(csv_reader)
        for row in csv_reader:
            yield list(map(float, row))

dataset_generator = read_dataset()

@app.route('/')
def index():
    return render_template('random.html')

@app.route('/anomaly')
def anomaly():
    return render_template("anomaly.html")


@app.route('/predict', methods=['POST'])
def predict():
    try:
      
        random_row = data.sample(n=1, random_state=random.seed()).values[0]
      
        
       
        CT_gas, V_gas, AP_gas, RH_gas, RPM_GAS_gas, EX_TEMP_gas, RPM_STEAM, STEAM_PRESSURE, WATER_FR, POWER_1, POWER_2 = random_row


        user_input_gas = [[CT_gas, V_gas, AP_gas, RH_gas, RPM_GAS_gas, EX_TEMP_gas]]
        user_input_steam = [[EX_TEMP_gas, RPM_STEAM, STEAM_PRESSURE, WATER_FR]]


        predicted_ep_gas = gas_model.predict(user_input_gas)
        predicted_ep_steam = steam_model.predict(user_input_steam)


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
            'predicted_ep_steam': float(predicted_ep_steam[0])
        }
        return jsonify(response_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_gas_custom', methods=['POST'])
def predict_gas_custom():
    try:
        data = request.json 
        ct = data['ct']
        v = data['v']
        ap = data['ap']
        rh = data['rh']
        rpm_gas = data['rpm_gas']
        ex_temp_gas = data['ex_temp_gas']

        input_data = np.array([ct, v, ap, rh, rpm_gas, ex_temp_gas]).reshape(1, -1)
        predicted_gas_ep = gas_model.predict(input_data)[0][0]

        predicted_gas_ep = float(predicted_gas_ep)

        return jsonify({'predicted_gas_ep': predicted_gas_ep})
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Return error message with 500 status code
    

@app.route('/predict_steam_custom', methods=['POST'])
def predict_steam_custom():
    try:
        data = request.json  
        ex_temp = data['ex_temp']
        rpm_steam = data['rpm_steam']
        steam_pressure = data['steam_pressure']
        water_fr = data['water_fr']

      
        input_data = np.array([ex_temp, rpm_steam, steam_pressure, water_fr]).reshape(1, -1)
        predicted_steam_ep = steam_model.predict(input_data)[0][0]

    
        predicted_steam_ep = float(predicted_steam_ep)

        return jsonify({'predicted_steam_ep': predicted_steam_ep})
    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/predict_random_gas', methods=['GET'])
def predict_random_gas_route():
    try:
        gas_output, gas_inputs = predict_random_gas(gas_model)
        return jsonify({'random_gas_ep': gas_output})
    except Exception as e:
        return jsonify({'error': str(e)}), 500 
    


@app.route('/predict_random_steam', methods=['GET'])
def predict_random_steam_route():
    try:
        steam_output, steam_inputs = predict_random_steam(steam_model)
        return jsonify({'random_steam_ep': steam_output})
    except Exception as e:
        return jsonify({'error': str(e)}), 500 

    
 


@app.route('/predict_anomaly', methods=['POST'])
def predict_anomaly():
    global dataset_generator
    try:
     
        try:
            row = next(dataset_generator)
        except StopIteration:
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
    app.run(debug=True)
