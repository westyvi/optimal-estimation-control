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
import scipy
import pandas
import csv

# Load the CSV file using numpy
data = np.genfromtxt('./fuel_data.csv', delimiter=',', dtype=None)

# Extract data into numpy arrays
time = data[:, 0]  # time
uk = data[:, 1]    # fuel flow meter sensor data 
yk = data[:, 2]    # fuel tank level sensor data 
fk = data[:, 3]    # true fuel remaining
bk = data[:, 4]    # true flow meter bias 

dt = 0.5 # seconds
A_line = 1 # cm^2
A_tank = 150 # cm^2
Q = np.diag([A_line**2 * dt**2, 0.1**2]) # cm^6, cm^2/s^2
R = 1**2 # cm
x0 = np.array([3000, 0]) # cm^3, cm/s
P0 = np.diag([10**2, 0.1**2]) # cm^6, cm^2/s^2

# implement state space 'motion' model for fuel system:
    # state vector: [fuel_remaining, flowRate_bias]
F = np.array([[1, A_line*dt],[0,1]])
G = np.array([-A_line*dt, 0])
H = np.array([A_tank**-1, 0])

filters = dict()
    
# steady-state Kalman Filter (SS_KF)

# standard discrete-time Kalman Filter (KF)
filters['KF']

# covariance intersection Kalman Filter (CI-KF)

# fixed-interval Kalman Smoother (FI_KS)

# commentary