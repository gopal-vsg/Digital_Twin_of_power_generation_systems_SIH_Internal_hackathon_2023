from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import random
import numpy as np


# from Combined_functions import process_data 
app = Flask(__name__)

# Load your machine learning models
gas_model = joblib.load('gas_model.joblib')
steam_model = joblib.load('steam_model.joblib')

# Load your dataset
data = pd.read_csv("datasets\Folds5x2_pp.csv")

# anomaly_data = pd.read_csv("Anamoly.csv")


@app.route('/')
def index():
    return render_template('random.html')



@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Randomly select a row from the dataset as input
        random_row = data.sample(n=1, random_state=random.seed()).values[0]

        CT_gas, V_gas, AP_gas, RH_gas, RPM_GAS_gas, EX_TEMP_gas, RPM_STEAM, STEAM_PRESSURE, WATER_FR, POWER_1, POWER_2 = random_row

        # Perform predictions using machine learning models for both gas and steam
        user_input_gas = [[CT_gas, V_gas, AP_gas, RH_gas, RPM_GAS_gas, EX_TEMP_gas]]
        user_input_steam = [[EX_TEMP_gas, RPM_STEAM, STEAM_PRESSURE, WATER_FR]]


        predicted_ep_gas = gas_model.predict(user_input_gas)
        predicted_ep_steam = steam_model.predict(user_input_steam)

        # Return predictions in JSON format with all necessary properties
        response = {
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
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    




@app.route('/predict_gas_custom', methods=['POST'])
def predict_gas_custom():
    try:
        data = request.json  # Assuming you send JSON data with user input
        ct = data['ct']
        v = data['v']
        ap = data['ap']
        rh = data['rh']
        rpm_gas = data['rpm_gas']
        ex_temp_gas = data['ex_temp_gas']

        # Convert input to NumPy array and make a prediction
        input_data = np.array([ct, v, ap, rh, rpm_gas, ex_temp_gas]).reshape(1, -1)
        predicted_gas_ep = gas_model.predict(input_data)[0][0]

        # Convert the predicted_gas_ep to a regular Python float
        predicted_gas_ep = float(predicted_gas_ep)

        return jsonify({'predicted_gas_ep': predicted_gas_ep})
    except Exception as e:
        return jsonify({'error': str(e)}), 500  # Return error message with 500 status code

# Route to predict steam for custom input
@app.route('/predict_steam_custom', methods=['POST'])

def predict_steam_custom():
    try:
        data = request.json  # Assuming you send JSON data with user input
        ex_temp = data['ex_temp']
        rpm_steam = data['rpm_steam']
        steam_pressure = data['steam_pressure']
        water_fr = data['water_fr']

        # Convert input to NumPy array and make a prediction
        input_data = np.array([ex_temp, rpm_steam, steam_pressure, water_fr]).reshape(1, -1)
        predicted_steam_ep = steam_model.predict(input_data)[0][0]

        # Convert the predicted_steam_ep to a regular Python float
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
        return jsonify({'error': str(e)}), 500  # Return error message with 500 status code


@app.route('/predict_random_steam', methods=['GET'])
def predict_random_steam_route():
     # Return error message with 500 status code



 if __name__ == '__main__':
    app.run(debug=True,port="5001")
