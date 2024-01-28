import pandas as pd
import joblib
import time
import tkinter as tk
from tkinter import messagebox
from threading import Thread

data = pd.read_csv("Anamoly.csv")
gas_model = joblib.load('gas_model.joblib')
steam_model = joblib.load('steam_model.joblib')

def update_labels(i):
    row = data.iloc[i]
    ct_entry.delete(0, tk.END)
    ct_entry.insert(0, row['CT'])
    v_entry.delete(0, tk.END)
    v_entry.insert(0, row['V'])
    ap_entry.delete(0, tk.END)
    ap_entry.insert(0, row['AP'])
    rh_entry.delete(0, tk.END)
    rh_entry.insert(0, row['RH'])
    rpm_gas_entry.delete(0, tk.END)
    rpm_gas_entry.insert(0, row['RPM_GAS'])
    ex_temp_entry.delete(0, tk.END)
    ex_temp_entry.insert(0, row['EX_TEMP'])
    rpm_steam_entry.delete(0, tk.END)
    rpm_steam_entry.insert(0, row['RPM_STEAM'])
    steam_pressure_entry.delete(0, tk.END)
    steam_pressure_entry.insert(0, row['STEAM_PRESSURE'])
    power1_entry.delete(0, tk.END)
    power1_entry.insert(0, row['POWER_1'])
    power2_entry.delete(0, tk.END)
    power2_entry.insert(0, row['POWER_2'])

    user_input_system = [row['CT'], row['V'], row['AP'], row['RH'], row['RPM_GAS'], row['EX_TEMP']]
    user_input_steam = [row['EX_TEMP'], row['RPM_STEAM'], row['STEAM_PRESSURE'], row['WATER_FR']]

    predicted_ep_gas = gas_model.predict([user_input_system])
    predicted_ep_steam = steam_model.predict([user_input_steam])

    predicted_power1_entry.delete(0, tk.END)
    predicted_power1_entry.insert(0, predicted_ep_gas[0])

    predicted_power2_entry.delete(0, tk.END)
    predicted_power2_entry.insert(0, predicted_ep_steam[0])

    time.sleep(1)

    if abs(predicted_ep_gas[0] - row['POWER_1']) > 150 or abs(predicted_ep_steam[0] - row['POWER_2']) > 100:
        messagebox.showwarning("Anomaly Detected", "Anomaly detected! Auto-shutdown initiated.")

def process_data():
    for i in range(len(data)):
        update_labels(i)
        root.update_idletasks()

def process_data_thread():
    thread = Thread(target=process_data)
    thread.start()

root = tk.Tk()
root.title("Anomaly Detection")

title_label = tk.Label(root, text="Anomaly Detection", font=("Helvetica", 16, "bold"))
title_label.grid(row=0, column=1, pady=(10, 10))  # Slightly reduced top padding

labels = ['CT', 'V', 'AP', 'RH', 'RPM_GAS', 'EX_TEMP', 'RPM_STEAM', 'STEAM_PRESSURE', 'POWER_1', 'POWER_2',
          'Predicted POWER1', 'Predicted POWER2']

tk.Label(root, text="").grid(row=0, column=0, pady=2)  # Reduced top padding
tk.Label(root, text="").grid(row=len(labels) + 1, column=0, pady=2)  # Reduced bottom padding

for i, label in enumerate(labels):
    tk.Label(root, text=label, anchor='e').grid(row=i+1, column=0, sticky='e', pady=2, padx=5)

ct_entry = tk.Entry(root, width=30)
v_entry = tk.Entry(root, width=30)
ap_entry = tk.Entry(root, width=30)
rh_entry = tk.Entry(root, width=30)
rpm_gas_entry = tk.Entry(root, width=30)
ex_temp_entry = tk.Entry(root, width=30)
rpm_steam_entry = tk.Entry(root, width=30)
steam_pressure_entry = tk.Entry(root, width=30)
power1_entry = tk.Entry(root, width=30)
power2_entry = tk.Entry(root, width=30)
predicted_power1_entry = tk.Entry(root, width=30)
predicted_power2_entry = tk.Entry(root, width=30)

ct_entry.grid(row=0, column=2, pady=(2, 5), ipadx=10, ipady=5, sticky='w')  # Reduced top padding
v_entry.grid(row=1, column=2, pady=5, ipadx=10, ipady=5, sticky='w')
ap_entry.grid(row=2, column=2, pady=5, ipadx=10, ipady=5, sticky='w')
rh_entry.grid(row=3, column=2, pady=5, ipadx=10, ipady=5, sticky='w')
rpm_gas_entry.grid(row=4, column=2, pady=5, ipadx=10, ipady=5, sticky='w')
ex_temp_entry.grid(row=5, column=2, pady=5, ipadx=10, ipady=5, sticky='w')
rpm_steam_entry.grid(row=6, column=2, pady=5, ipadx=10, ipady=5, sticky='w')
steam_pressure_entry.grid(row=7, column=2, pady=5, ipadx=10, ipady=5, sticky='w')
power1_entry.grid(row=8, column=2, pady=5, ipadx=10, ipady=5, sticky='w')
power2_entry.grid(row=9, column=2, pady=5, ipadx=10, ipady=5, sticky='w')
predicted_power1_entry.grid(row=10, column=2, pady=5, ipadx=10, ipady=5, sticky='w')
predicted_power2_entry.grid(row=11, column=2, pady=(5, 20), ipadx=10, ipady=5, sticky='w')  # Reduced bottom padding

process_button = tk.Button(root, text="Process Data", command=process_data_thread, width=40)
process_button.grid(row=len(labels) + 2, columnspan=3, pady=(5, 2), ipadx=10, ipady=5, sticky='w')  # Reduced top padding

root.grid_rowconfigure(0, weight=1)
root.grid_rowconfigure(len(labels) + 3, weight=1)
root.grid_columnconfigure(1, weight=1)

root.mainloop()