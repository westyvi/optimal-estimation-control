#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AEM566 - Optimal Estimation and Control
Project 3:
    
    Nonlinear Least Squares Battery Parameter Estimation 

Commentary and initial assignment contained in project folder
    
Written by Joey Westermeyer 2024
    
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy
import csv
import os
plot_dir = os.path.join(os.path.dirname(__file__), 'plots')
os.makedirs(plot_dir, exist_ok=True)

def load_pulse_test_data(file_path):
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
times, voltages, currents = load_pulse_test_data('./pulse_discharge_test_data.csv')
times -= times[0]

# Integrate current to get state of charge (Q); assume Q[0] = 0
Q = np.copy(times)
Q[0] = 0
time_step = times[1]-times[0]
Q = np.cumsum(currents) * - time_step # in joules
Q -= Q[-1] # find total charge assuming ending Q=0
SOC = Q/Q[0] # define SOC assuming Q[0] corresponds to 100% SOC

# break up discharge data into rest periods
rest_period = []
amp_threshold = 0.1 # determines whether in a rest period or not
start = 0
end = 0
i = 0
while i < currents.size:
    if currents[i] < amp_threshold:
        start = i-1 # want to include last discharge data point
        for end in range(i,currents.size):
            if currents[end] > amp_threshold:
                break
        i = end
        rest_period.append({'current':currents[start:end], 'voltage':voltages[start:end], 'time':times[start:end]})
    i += 1
del rest_period[0] # first rest period isn't a voltage recovery period; it is pre test-start

# calculate optimal battery parameters for each rest period using nonlinear least squares
num_exponentials = 3 # number of RC exponential terms to approximate
fig, ax = plt.subplots()
for idx, period in enumerate(rest_period):
    sol = []
    cost = []
    residual = []
    for n in range(1,num_exponentials+1):
        i_dis = period['current'][0]
        R0 = (period['voltage'][1] - period['voltage'][0])/i_dis
        OCV_estimate = period['voltage'][-1]
        R1_estimate = 0.95*(OCV_estimate - period['voltage'][1])/i_dis
        tau_estimate = np.argmax(period['voltage'] >= 0.95*(OCV_estimate - period['voltage'][1]) + period['voltage'][1])
        x0 = np.array([OCV_estimate, R1_estimate, tau_estimate])
        x0 = np.append(x0, np.ones(2*(n-1)))
        
        def fun(x,t):
            res = x[0]
            for i in range (0,n):
                if x[2+2*i]==0:
                    continue
                res -= i_dis*x[1+2*i]*np.exp(-t/x[2+2*i])
            return res
        
        def residual_fun(x, t, V):
            return fun(x,t) - V
        
        # compute optimal parameters using nonlinear least squares
        result = scipy.optimize.least_squares(
            residual_fun, 
            x0, 
            loss='soft_l1',
            bounds = (x0*0, np.inf), # restrict all parameters to positive or zero values
            args=(period['time'][1:]-period['time'][1], period['voltage'][1:]), # cut pre-recovery [0] period data point
            )
        
        # insert calculated R0 parameter to solution parameter list
        sol.append(np.insert(result.x, 1, R0))
        cost.append(result.cost*2) # scipy calculates cost function as 0.5*residual**2
        
        # plot data and fit for this rest period
        if idx == 0:  # Only plot for the chosen rest period (update idx as needed)
            if n == 1:
                ax.plot(period['time'][1:], period['voltage'][1:], label='Measured OCV')
                ax.set(xlabel='Time (s)', ylabel='Voltage (V)', title='Measured OCV During One Pulse Discharge')
            ax.plot(
                period['time'][1:],
                fun(result.x, period['time'][1:] - period['time'][1]),
                label=f'Model fit ({n} RC term{"s" if n > 1 else ""})'
            )
        ax.legend()

        
        residual_error = np.sum((fun(result.x,period['time'][1:-1]-period['time'][1]) - period['voltage'][1:-1])**2)/2
        residual.append(residual_error)
        
    period['optimal_params'] = sol
    period['residuals'] = np.array(residual)
    period['cost'] = cost
    i += 1
fig.savefig(os.path.join(plot_dir, 'single_period_fit.png'), bbox_inches='tight')
plt.close(fig)

# pull data from each period for plotting
residuals = np.zeros((len(rest_period), num_exponentials))
OCV = residuals.copy()
R1 = residuals.copy()
T1 = residuals.copy()
T2 = np.zeros((len(rest_period), num_exponentials-1))
R2 = T2.copy()
T3 = np.zeros((len(rest_period), num_exponentials-2))
R3 = T3.copy()
R0 = np.array([period_data['optimal_params'][0][1] for period_data in rest_period])
sub_times = np.array([period_data['time'][1] for period_data in rest_period])
time_indices = np.where(np.isin(times, sub_times))
SOCs = SOC[time_indices]
for i in range(0,num_exponentials):
    residuals[:,i] = np.array([period_data['residuals'][i] for period_data in rest_period])
    OCV[:,i] = np.array([period_data['optimal_params'][i][0] for period_data in rest_period])
    R1[:,i] = np.array([period_data['optimal_params'][i][2] for period_data in rest_period])
    T1[:,i] = np.array([period_data['optimal_params'][i][3] for period_data in rest_period])
    if i > 0:
        R2[:,i-1] = np.array([period_data['optimal_params'][i][4] for period_data in rest_period])
        T2[:,i-1] = np.array([period_data['optimal_params'][i][5] for period_data in rest_period])
    if i > 1:
        R3[:,i-2] = np.array([period_data['optimal_params'][i][6] for period_data in rest_period])
        T3[:,i-2] = np.array([period_data['optimal_params'][i][7] for period_data in rest_period])

# Plot measured OCV (voltage) vs time during pulse discharge test
fig, ax = plt.subplots()
ax.plot(times, voltages, label='Measured Voltage', color='black', linewidth=1)
ax.set(xlabel='Time (s)', ylabel='Voltage (V)', title='Measured OCV During Pulse Discharge Test')
ax.legend()
fig.savefig(os.path.join(plot_dir, 'ocv_vs_time.png'), bbox_inches='tight')
plt.close(fig)

# Residuals vs SOC
fig, ax = plt.subplots()
for i in range(num_exponentials):
    ax.plot(SOCs, residuals[:,i], label=f'RC model with {i+1} exponential terms')
ax.set(xlabel='SOC', ylabel='Residuals', title='Residuals vs SOC')
ax.legend()
fig.savefig(os.path.join(plot_dir, 'residuals_vs_soc.png'), bbox_inches='tight')
plt.close(fig)

# OCV vs SOC
fig, ax = plt.subplots()
for i in range(num_exponentials):
    ax.plot(SOCs, OCV[:,i], label=f'RC model with {i+1} exponential terms')
ax.set(xlabel='SOC', ylabel='OCV', title='OCV vs SOC')
ax.legend()
fig.savefig(os.path.join(plot_dir, 'ocv_vs_soc.png'), bbox_inches='tight')
plt.close(fig)

# R1/T1 vs SOC
fig, ax = plt.subplots()
for i in range(num_exponentials):
    ax.plot(SOCs, R1[:,i], label=f'R1, RC model with {i+1} exponential terms')
ax.set(xlabel='SOC', ylabel='R1', title='R1 vs SOC')
ax.legend()
fig.savefig(os.path.join(plot_dir, 'r1_vs_soc.png'), bbox_inches='tight')
plt.close(fig)

fig, ax = plt.subplots()
for i in range(num_exponentials):
    ax.plot(SOCs, T1[:,i], label=f'T1, RC model with {i+1} exponential terms')
ax.set(xlabel='SOC', ylabel='T1', title='T1 vs SOC')
ax.legend()
fig.savefig(os.path.join(plot_dir, 't1_vs_soc.png'), bbox_inches='tight')
plt.close(fig)

# R2/T2 and R3/T3 if present
if num_exponentials > 1:
    fig, ax = plt.subplots()
    for i in range(1, num_exponentials):
        ax.plot(SOCs, R2[:,i-1], label=f'R2, RC model with {i+1} exponential terms')
    ax.set(xlabel='SOC', ylabel='R2', title='R2 vs SOC')
    ax.legend()
    fig.savefig(os.path.join(plot_dir, 'r2_vs_soc.png'), bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots()
    for i in range(1, num_exponentials):
        ax.plot(SOCs, T2[:,i-1], label=f'T2, RC model with {i+1} exponential terms')
    ax.set(xlabel='SOC', ylabel='T2', title='T2 vs SOC')
    ax.legend()
    fig.savefig(os.path.join(plot_dir, 't2_vs_soc.png'), bbox_inches='tight')
    plt.close(fig)

if num_exponentials > 2:
    fig, ax = plt.subplots()
    for i in range(2, num_exponentials):
        ax.plot(SOCs, R3[:,i-2], label=f'R3, RC model with {i+1} exponential terms')
    ax.set(xlabel='SOC', ylabel='R3', title='R3 vs SOC')
    ax.legend()
    fig.savefig(os.path.join(plot_dir, 'r3_vs_soc.png'), bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots()
    for i in range(2, num_exponentials):
        ax.plot(SOCs, T3[:,i-2], label=f'T3, RC model with {i+1} exponential terms')
    ax.set(xlabel='SOC', ylabel='T3', title='T3 vs SOC')
    ax.legend()
    fig.savefig(os.path.join(plot_dir, 't3_vs_soc.png'), bbox_inches='tight')
    plt.close(fig)
