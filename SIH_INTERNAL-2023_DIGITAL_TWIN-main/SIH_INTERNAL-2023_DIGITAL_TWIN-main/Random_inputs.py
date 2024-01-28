import numpy as np
import time
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import threading




# Load data and perform necessary preprocessing
data = pd.read_csv('datasets\Thermal_powerplant.csv')
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
sc = StandardScaler()
x = sc.fit_transform(x)
lr = LinearRegression()
lr.fit(x, y)
parameter_names = data.columns[:-1]

def display_random_output():
    while True:
        random_index = random.randint(0, len(x) - 1)
        random_input = x[random_index]
        parameter_value_strings = [f'"{param}" = {value:.2f}' for param, value in zip(parameter_names, random_input)]
        output_string = " | ".join(parameter_value_strings)

        actual_output = y[random_index]
        random_input = random_input.reshape(1, -1)
        predicted_output = lr.predict(random_input)[0]

        print(
            f"\r{output_string} | Predicted Output = {predicted_output:.2f} | Actual Output = {actual_output:.2f}",
            end="")
        time.sleep(1)
# Start the thread for random output display
random_output_thread = threading.Thread(target=display_random_output)
random_output_thread.daemon = True
random_output_thread.start()


while True:
    pass
