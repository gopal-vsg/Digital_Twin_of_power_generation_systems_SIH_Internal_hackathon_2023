import numpy as np
import time
import random
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

data = pd.read_csv('Thermal_powerplant.csv')
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
sc = StandardScaler()
x = sc.fit_transform(x)
lr = LinearRegression()
lr.fit(x, y)

inlet_steam_temperature = float(input("Enter Inlet Steam Temperature: "))
outlet_steam_temperature = float(input("Enter Outlet Steam Temperature (C): "))
steam_pressure_at_boiler_outlet = float(input("Enter Steam Pressure at Boiler Outlet: "))
feedwater_temperature = float(input("Enter Feedwater Temperature: "))
steam_flow_rate = float(input("Enter Steam Flow Rate (Kg/h): "))
feedwater_flow_rate = float(input("Enter Feedwater Flow Rate (Kg/h): "))
boiler_efficiency = float(input("Enter Boiler Efficiency (%): "))
turbine_efficiency = float(input("Enter Turbine Efficiency (%): "))
steam_quantity = float(input("Enter Steam Quantity (%): "))
co2_emission_rate = float(input("Enter CO2 Emission Rate (Kg/h): "))
boiler_pressure_drop = float(input("Enter Boiler Pressure Drop (bar): "))
condenser_cooling_water_flow_rate = float(input("Enter Condenser Cooling Water Flow Rate (m^3/h): "))
steam_turbine_rpm = float(input("Enter Steam Turbine RPM: "))
generator_efficiency = float(input("Enter Generator Efficiency: "))

user_inputs = np.array([[
    inlet_steam_temperature,
    outlet_steam_temperature,
    steam_pressure_at_boiler_outlet,
    feedwater_temperature,
    steam_flow_rate,
    feedwater_flow_rate,
    boiler_efficiency,
    turbine_efficiency,
    steam_quantity,
    co2_emission_rate,
    boiler_pressure_drop,
    condenser_cooling_water_flow_rate,
    steam_turbine_rpm,
    generator_efficiency
]])

predicted_power_output = lr.predict(sc.transform(user_inputs))

print("Predicted Electrical Power Output (MW):", predicted_power_output[0])