import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import random
import joblib
import matplotlib.pyplot as plt
import mpld3

# Load the data for training
data = pd.read_csv("Folds5x2_pp.csv")


# Gas Model
def train_gas_model():
    X = data[['CT', 'V', 'AP', 'RH', "RPM_GAS", "EX_TEMP"]]
    y = data['POWER_1']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Compile the model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(6,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer with 1 neuron for regression
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

    return model


def predict_gas_ep(model):
    temperature = float(input("Enter Temperature (AT): "))
    vacuum = float(input("Enter Exhaust Vacuum (V): "))
    pressure = float(input("Enter Ambient Pressure (AP): "))
    humidity = float(input("Enter Relative Humidity (RH): "))
    RPM_GAS = float(input("Enter RMP GAS (RPM_GAS): "))
    EX_TEMP = float(input("Enter EX TEMP (EX_TEMP): "))

    user_input = [[temperature, vacuum, pressure, humidity, RPM_GAS, EX_TEMP]]

    predicted_ep = model.predict(user_input)

    print(f"Predicted EP: {predicted_ep[0][0]}")


# Steam Model
def train_steam_model():
    X = data[['EX_TEMP', 'RPM_STEAM', 'STEAM_PRESSURE', 'WATER_FR']]
    y = data['POWER_2']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Compile the model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Output layer with 1 neuron for regression
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

    return model

try:
    gas_model = joblib.load("gas_model.joblib")
    steam_model = joblib.load("steam_model.joblib")
    print("Loaded pre-trained models.")
except FileNotFoundError:
    print("Pre-trained models not found. Training new models.")
    gas_model = train_gas_model()
    steam_model = train_steam_model()
    joblib.dump(gas_model, "gas_model.joblib")
    joblib.dump(steam_model, "steam_model.joblib")
    
def predict_steam_ep(model):
    EX_TEMP = float(input("Enter Exhaust Temperature (EX_TEMP): "))
    RPM_STEAM = float(input("Enter RPM STEAM (RP_STEAM): "))
    STEAM_PRESSURE = float(input("Enter STEAM PRESSURE (STEAM_PRESSURE): "))
    WATER_FR = float(input("Enter WATER FR (WATER_FR): "))

    user_input = [[EX_TEMP, RPM_STEAM, STEAM_PRESSURE, WATER_FR]]

    predicted_ep = model.predict(user_input)

    print(f"Predicted EP: {predicted_ep[0][0]}")


# Random Prediction
def predict_random():
    random_row = data.sample(n=1, random_state=random.seed()).values[0]

    CT, V, AP, RH, RPM_GAS, EX_TEMP, RPM_STEAM, STEAM_PRESSURE, WATER_FR = random_row[:9]

    user_input_gas = [[CT, V, AP, RH, RPM_GAS, EX_TEMP]]
    user_input_steam = [[EX_TEMP, RPM_STEAM, STEAM_PRESSURE, WATER_FR]]

    predicted_ep_gas = gas_model.predict(user_input_gas)
    predicted_ep_steam = steam_model.predict(user_input_steam)

    print(f"Predicted Gas EP: {predicted_ep_gas[0][0]:.2f} MW")
    print(f"Predicted Steam EP: {predicted_ep_steam[0][0]:.2f} MW")


# Start and stop auto prediction
def start_auto_predict():
    global auto_predict
    auto_predict = True
    predict_random()


def stop_auto_predict():
    global auto_predict
    auto_predict = False


auto_predict = False


# ADD the below code as another function
def process_data():
    anomaly_data = pd.read_csv("Anamoly.csv")
    for i in range(len(anomaly_data)):
        row = anomaly_data.iloc[i]

        user_input_system = [row['CT'], row['V'], row['AP'], row['RH'], row['RPM_GAS'], row['EX_TEMP']]
        user_input_steam = [row['EX_TEMP'], row['RPM_STEAM'], row['STEAM_PRESSURE'], row['WATER_FR']]

        predicted_ep_gas = gas_model.predict([user_input_system])
        predicted_ep_steam = steam_model.predict([user_input_steam])

        if abs(predicted_ep_gas[0] - row['POWER_1']) > 100 or abs(predicted_ep_steam[0] - row['POWER_2']) > 100:
            print("Anomaly Detected at row", i)
            print("Actual POWER_1:", row['POWER_1'])
            print("Predicted POWER_1:", predicted_ep_gas[0])
            print("Actual POWER_2:", row['POWER_2'])
            print("Predicted POWER_2:", predicted_ep_steam[0])
            print("Auto-shutdown initiated.")
        else:
            print("No anomaly detected at row", i)


# Switch Case function
def switch_case(option):
    if option == 1:
        predict_gas_ep(gas_model)
    elif option == 2:
        predict_steam_ep(steam_model)
    elif option == 3:
        predict_random()
    elif option == 4:
        process_data()
    else:
        print("Invalid option. Please choose a valid option.")


# Training the models
gas_model = train_gas_model()
steam_model = train_steam_model()

# Main loop
while True:
    print("\nOptions:")
    print("1. Predict Gas")
    print("2. Predict Steam")
    print("3. Predict Random")
    print("4. Process Anomaly Data")
    print("5. Quit")

    option = int(input("Enter your choice (1-5): "))

    if option == 5:
        break

    switch_case(option) 
    

