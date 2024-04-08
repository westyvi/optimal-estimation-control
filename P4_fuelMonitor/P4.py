#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AEM566 - Optimal Estimation and Control
Project 4:
    
    Kalman Filtering for fuel estimation 

Commentary and initial assignment contained in project folder
    
Written by Joey Westermeyer 2024

notes:
    
"""

import numpy as np
from matplotlib import pyplot as plt
import csv
import P4_classes

# Load the CSV file using numpy
csv_data = np.genfromtxt('./fuel_data.csv', delimiter=',', dtype=None)

# Extract data into numpy arrays
time = csv_data[:, 0]  # time
uk = csv_data[:, 1]    # fuel flow meter sensor data 
yk = csv_data[:, 2]    # fuel tank level sensor data 
fk = csv_data[:, 3]    # true fuel remaining
bk = csv_data[:, 4]    # true flow meter bias 

# define model parameters
dt = 0.5 # seconds
A_line = 1 # cm^2
A_tank = 150 # cm^2
Q = np.diag([A_line**2 * dt**2, 0.1**2]) # cm^6, cm^2/s^2
R = 1**2 # cm
x0 = np.array([3000, 0]) # cm^3, cm/s
P0 = np.diag([10**2, 0.1**2]) # cm^6, cm^2/s^2

# define state space model for fuel system:
# state vector: [fuel_remaining, flowRate_bias]
F = np.array([[1, A_line*dt],[0,1]])
G = np.array([-A_line*dt, 0])
H = np.array([A_tank**-1, 0])

# dict to log results of running filters
filters_log = dict()
template = dict()
template['x'] = np.zeros((time.size, x0.size))
template['P'] = np.zeros((time.size, P0.shape[0], P0.shape[1]))
template['x'][0,:] = x0
template['P'][0,:,:] = P0
filters_log['KF'] = template
filters_log['SSKF'] = template
filters_log['CIKF'] = template
filters_log['FIKS'] = template
    
# steady-state Kalman Filter (SS_KF)

# standard discrete-time Kalman Filter (KF)
KF = P4_classes.KF(F, G, Q, H, R)
filter_type = 'KF'
for i in range(0,time.size-1):
    # could make faster by not looking up data in log, but this more readable
    x_apriori, P_apriori = KF.predict(filters_log[filter_type]['x'][i,:], filters_log[filter_type]['P'][i,:,:], dt, uk[i])
    x_posteriori, P_posteriori = KF.correct(yk[i+1], x_apriori, P_apriori)
    filters_log[filter_type]['x'][i+1,:] = x_posteriori
    filters_log[filter_type]['P'][i+1,:,:] = P_posteriori

# covariance intersection Kalman Filter (CI-KF)
# bound omega <= 0.95 for search stability

# fixed-interval Kalman Smoother (FI_KS)

# plotting
def plot(filter_log, filter_string):
    # Plot the posterior state estimates versus the true states for the fuel remaining.
    fig, ax = plt.subplots()
    ax.plot(time, filter_log['x'][:,0], c='red', label='estimated')
    ax.plot(time, fk, c='black', label='truth')
    ax.set(xlabel = 't, s', ylabel = 'Fuel Remaining, cm^3',
          title = 'Estimated Fuel Remaining vs time, ' + filter_string)
    ax.legend()
    plt.grid(True)
    
    # Plot the posterior state estimates versus the true states for the flow meter bias.
    fig, ax = plt.subplots()
    ax.plot(time, filter_log['x'][:,1], c='red', label='estimated')
    ax.plot(time, bk, c='black', label='truth')
    ax.set(xlabel = 't, s', ylabel = 'Flowmeter Bias, cm',
          title = 'Estimated Flowmeter Bias vs time, ' + filter_string)
    ax.legend()
    plt.grid(True)
    
    # Plot the posterior state estimate errors and the ±2𝜎 from the diagonals of the steady-state
    # posterior covariance for the fuel remaining.
    fig, ax = plt.subplots()
    ax.plot(time, filter_log['x'][:,0]-fk, c='red', label=filter_string)
    ax.fill_between(time, -filter_log['P'][:,0,0], filter_log['P'][:,0,0], color='purple', alpha=0.5, label='2-sigma covariance bound')
    ax.set(xlabel = 't, s', ylabel = 'Fuel Remaining, cm^3',
          title = 'Posterior State Estimate Errors and estimated 2-sigma bounds vs time, ' + filter_string)
    ax.legend()
    plt.grid(True)
    
    # Plot the posterior state estimate errors and the ±2𝜎 from the diagonals of the steady-state
    # posterior covariance for the flow meter bias.
    fig, ax = plt.subplots()
    ax.plot(time, filter_log['x'][:,1]-bk, c='red', label=filter_string)
    ax.fill_between(time, -filter_log['P'][:,1,1], filter_log['P'][:,1,1], color='purple', alpha=0.5, label='2-sigma covariance bound')
    ax.set(xlabel = 't, s', ylabel = 'Flowmeter Bias, cm',
          title = 'Posterior State Estimate Errors and estimated 2-sigma bounds vs time, ' + filter_string)
    ax.legend()
    plt.grid(True)
    
plot(filters_log['KF'], 'KF')
# extra: plot kalman gains vs SSKF gains
# commentary