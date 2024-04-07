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
filters_log['KF'] = template
filters_log['SSKF'] = template
filters_log['CIKF'] = template
filters_log['FIKS'] = template
    
# steady-state Kalman Filter (SS_KF)

# standard discrete-time Kalman Filter (KF)
KF = P4_classes.KF(F, G, Q, H, R)


# covariance intersection Kalman Filter (CI-KF)
# bound omega <= 0.95 for search stability

# fixed-interval Kalman Smoother (FI_KS)

# commentary