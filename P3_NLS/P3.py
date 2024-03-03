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
        time_last = 0
        day_count = 0
        for row in csv_reader:
            # Assuming the CSV columns are in the order: time, voltage, current
            time_str, voltage_str, current_str = row
            
            # Convert time string to a time object
            time_parts = time_str.split(':')
            hours, minutes, seconds = map(int, time_parts)
            time_seconds = hours * 3600 + minutes * 60 + seconds + day_count * 24 * 60 * 60
            # account for change of day, e.g. hour count resets 24 -> 0
            if time_last>time_seconds:
                day_count += 1
                time_seconds += 24*60*60
            times.append(time_seconds)
            time_last = time_seconds
            
            # Convert voltage and current strings to floats
            voltages.append(float(voltage_str))
            currents.append(float(current_str))
            
        # Convert lists to NumPy arrays
        times = np.array(times)
        voltages = np.array(voltages)
        currents = np.array(currents)

    return times, voltages, currents

# load data in to arrays
times, voltages, currents = load_data_from_csv('./pulse_discharge_test_data.csv')
times -= times[0]

# Integrate current to get state of charge (Q); assume Q[0] = 0
Q = np.copy(times)
Q[0] = 0
time_step = times[1]-times[0]
Q = np.cumsum(currents) * - time_step # in joules
Q -= Q[-1] # find total charge assuming ending Q=0
SOC = Q/Q[0] # define SOC assuming Q[0] corresponds to 100% SOC

# plot Q vs time
fig, ax = plt.subplots()
ax.plot(times, SOC*100)
ax.set(xlabel = 'Time, s', ylabel = 'State of Charge, percent',
      title = 'State of Charge vs Time')
ax.grid(True)

# break up discharge data into rest periods
rest_period = []
amp_threshold = 0.1 # determines whether in a rest period or not
start = 0
end = 0
i = 0
while i < currents.size:
    # if found the start of a rest period
    if currents[i] < amp_threshold:
        start = i
        for end in range(i,currents.size):
            if currents[end] > amp_threshold:
                break
        i = end
        rest_period.append(currents[start:end])
    i += 1
            









# cracked multi case plotting function if you want to use it
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