#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AEM566 - Optimal Estimation and Control
Project 3:
    
    Nonlinear Least Squares Battery Parameter Estimation 

Contents:
    - 
    - Plot results
    
Commentary and initial assignment contained in project folder
    
Written by Joey Westermeyer 2024

notes:
    
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy
import pandas
import csv

def load_data_from_csv(file_path):
    times = []
    voltages = []
    currents = []

    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Skip header if present
        for row in csv_reader:
            # Assuming the CSV columns are in the order: time, voltage, current
            time_str, voltage_str, current_str = row
            
            # Convert time string to a time object
            time_parts = time_str.split(':')
            hours, minutes, seconds = map(int, time_parts)
            time_seconds = hours * 3600 + minutes * 60 + seconds
            times.append(time_seconds)
            
            # Convert voltage and current strings to floats
            voltages.append(float(voltage_str))
            currents.append(float(current_str))

    return times, voltages, currents

# Example usage:
times, voltages, currents = load_data_from_csv('./pulse_discharge_test_data.csv')


'''# %%plot results
t = np.linspace(0, simTime, numpts)
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
figs = [fig1, fig2, fig3]
axes = [ax1, ax2, ax3]
xyz_string_list = ['x', 'y', 'z']

# loop through each simulation case 
for sim_case in reversed(sim_runs):
    # loop through each data set (vx, vy, and vz)
    for i, ax in enumerate(axes):
        ax.plot(t, sim_runs[sim_case]['uvw_gust'][i,:], sim_runs[sim_case]['color'], label=sim_case)
        ax.set(xlabel = 't, s', ylabel = 'v, ft/s',
              title = xyz_string_list[i] + ' gust vs time')
        ax.legend()
        ax.grid(True)

    fig, ax = plt.subplots()
    ax.plot(sim_runs[sim_case]['uvw_gust'][1,:], sim_runs[sim_case]['uvw_gust'][2,:], 'b')
    ax.set(xlabel = 't, s', ylabel = 'v, ft/s',
          title = 'y vs z gust velocities, ' + sim_case + ' (see commentary)')
    ax.grid(True)
'''