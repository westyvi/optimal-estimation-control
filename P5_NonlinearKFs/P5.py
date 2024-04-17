#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AEM566 - Optimal Estimation and Control
Project 5:
    
    Nonlinear Kalman Filtering for ballistic altitude estimation

initial assignment contained in project folder
    
Written by Joey Westermeyer 2024
    
"""

import numpy as np
from matplotlib import pyplot as plt
import csv
import P5_classes
import copy

# Load the CSV file using numpy
csv_data = np.genfromtxt('./altimeter_data.csv', delimiter=',', dtype=None)

# Extract data into numpy arrays
time = csv_data[:, 0]  # time
uk = csv_data[:, 1]    # fuel flow meter sensor data 
yk = csv_data[:, 2]    # fuel tank level sensor data 
fk = csv_data[:, 3]    # true fuel remaining
bk = csv_data[:, 4]    # true flow meter bias 

# define model parameters
dt = 0.5 # seconds
rho0 = 0.0765 # lb/ft3
g0 = 32.2 # ft/s2
h_rho = 30000 # ft
R_E = 20902260 #ft
d = 100000 # ft
Q = np.diag([10**2, 10**2, 0.05**2])
R = 100**2 # ft2
x0 = np.array([400000, -2000, 20]) # ft, ft/s, lb/ft2
P0 = np.diag([100**2, 10**2, 1**2]) # ft2, ft2/s2, 

# define state space model for fuel system:
# state vector: [altitude, verticalSpeed, ballistic coefficient]
def propogate_state(x_hat):
    f0 = lambda x_hat: x_hat[0] + dt*x_hat[1]
    f1 = lambda x_hat: x_hat[1] + dt*(rho0*x_hat[1]**2/2/x_hat[2]*np.exp(-x_hat[0]/h_rho) - g0*(R_E/(R_E + x_hat[0]))**2)
    f2 = lambda x_hat: x_hat[2]
    return [f0, f1, f2]

i = 1
def measure(x_hat):
    pass

# dict to log results of running filters
filters_log = dict()
template = dict()
template['x'] = np.zeros((time.size, x0.size))
template['P'] = np.zeros((time.size, P0.shape[0], P0.shape[1]))
template['x'][0,:] = x0
template['P'][0,:,:] = P0
filters_log['EKF'] = copy.deepcopy(template)
filters_log['UKF'] = copy.deepcopy(template)
filters_log['BSKF'] = copy.deepcopy(template)
filters_log['FIKS'] = copy.deepcopy(template)

# add gain logging for EKF, UKF
filters_log['EKF']['gain'] = np.zeros((time.size-1, 2))
filters_log['UKF']['gain'] = np.zeros((time.size-1, 2))

# define function to run filters and log results for data in csv
def runFilter(filter_object, filter_log, log_gain = False):
    for i in range(0,time.size-1):
        # could make faster by not looking up data in log, but this more readable
        x_apriori, P_apriori = filter_object.predict(filter_log['x'][i,:], filter_log['P'][i,:,:], dt, uk[i])
        x_posteriori, P_posteriori = filter_object.correct(yk[i+1], x_apriori, P_apriori)
        filter_log['x'][i+1,:] = x_posteriori
        filter_log['P'][i+1,:,:] = P_posteriori
        if log_gain:
            filter_log['gain'][i, :] = filter_object.K

# plotting function
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
    ax.fill_between(time, -2*np.sqrt(filter_log['P'][:,0,0]), 2*np.sqrt(filter_log['P'][:,0,0]), color='purple', alpha=0.5, label='2-sigma covariance bound')
    ax.set(xlabel = 't, s', ylabel = 'Fuel Remaining, cm^3',
          title = 'Posterior State Estimate Errors and estimated 2-sigma bounds vs time, ' + filter_string)
    ax.legend()
    plt.grid(True)
    
    # Plot the posterior state estimate errors and the ±2𝜎 from the diagonals of the steady-state
    # posterior covariance for the flow meter bias.
    fig, ax = plt.subplots()
    ax.plot(time, filter_log['x'][:,1]-bk, c='red', label=filter_string)
    ax.fill_between(time, -2*np.sqrt(np.abs(filter_log['P'][:,1,1])), 2*np.sqrt(np.abs(filter_log['P'][:,1,1])), color='purple', alpha=0.5, label='2-sigma covariance bound')
    ax.set(xlabel = 't, s', ylabel = 'Flowmeter Bias, cm',
          title = 'Posterior State Estimate Errors and estimated 2-sigma bounds vs time, ' + filter_string)
    ax.legend()
    plt.grid(True)
    

# standard discrete-time Kalman Filter (EKF)
EKF = P4_classes.EKF(F, G, Q, H, R)
runFilter(EKF, filters_log['EKF'], True)
plot(filters_log['EKF'], 'EKF')

# steady-state Kalman Filter (SS_KF)
UKF = P4_classes.UKF(F, G, Q, H, R)
runFilter(UKF, filters_log['UKF'], True)
plot(filters_log['UKF'], 'UKF')

# plot kalman gains vs UKF gains
fig, ax = plt.subplots()
ax.plot(time[1:], filters_log['EKF']['gain'][:,0], c='red', label='EKF fuel remaining gain')
ax.plot(time[1:], filters_log['EKF']['gain'][:,1], c='black', label='EKF flowmeter bias gain')
ax.plot(time[1:], filters_log['UKF']['gain'][:,0], c='blue', label='UKF fuel remaining gain')
ax.plot(time[1:], filters_log['UKF']['gain'][:,1], c='green', label='UKF flowmeter bias gain')
ax.set(xlabel = 't, s', ylabel = 'Kalman gains, cm^3',
      title = 'UKF vs EKF Kalman gains')
ax.legend()
plt.grid(True)

# covariance intersection Kalman Filter (CI-EKF)
# bound omega <= 0.95 for search stability
BSKF = P4_classes.BSKF(F, G, Q, H, R)
runFilter(BSKF, filters_log['BSKF'])
plot(filters_log['BSKF'], 'BSKF')

# fixed-interval Kalman Smoother (FI_KS)
FIKS = P4_classes.FIKS(F, G, Q, H, R)
filters_log['FIKS'] = copy.deepcopy(filters_log['UKF'])
for i in range(time.size-2,-1,-1):
    # could make faster by not looking up data in log, but this more readable
    #x_plus1_posteriori, x_posteriori, u, P_plus1_posteriori, P_posteriori)
    x_smoothed, P_smoothed = FIKS.smooth(filters_log['FIKS']['x'][i+1,:], filters_log['UKF']['x'][i,:], uk[i], filters_log['FIKS']['P'][i+1,:,:], filters_log['UKF']['P'][i,:,:])
    filters_log['FIKS']['x'][i,:] = x_smoothed
    filters_log['FIKS']['P'][i,:,:] = P_smoothed
plot(filters_log['FIKS'], 'FIKS')

# commentary
'''As can be seen by the plots produced, the kalman filtering framework provides multiple
ways to estimate states via sensor fusion. As implemented, the base kalman filter struggles to 
track the true fuel remaining and flowmeter bias. While it appears as though this is caused 
by too low a process noise matrix, leading to too much trust in the model 
(shown by small covariance and out of bounds errors), the UKF child class
performs much better with all the same equations save for the kalman gain calculation, which is constant
and derived from the LQE DARE problem for the UKF. This is further supported by the kalman gain plot, which 
shows that the EKF does not converge to the UKF gains. I've checked the standard kalman filter gain calculation,
tried different expressions, and re-coded the function but haven't found the culprit for this behavior yet. 
However, in a normal scenario the EKF should converge to the UKF and have the advantage of being able to 
calculate the kalman gain adaptively online, leading to a more flexible and adaptive, while very similar, 
 solution as compared to the EKF. It can also be seen that the UKF approximates the covariances 
 of the states well, with the 95% bounds on both fuel remaining and meter bias containing the errors about 
 95 percent of the time. 
 
 Moving on, the CI-EKF can be seen to have worse performance, as it 
 struggles to fix steady state fuel remaining value with overly-small state covariances, then suddenly
 blows up in terms of covariance and erratically oscillates around the true states, seemingly 
 only chasing the measured values, discarding the prediction. This leads to very large errors, oscillations in 
 state, and covariances for most of the data window. So while the CI-EKF may be useful in 
 sensor fusion of unknown correlation or in different contexts, it appears to be a poor choice for 
 fusing this data.
 
 Finally, the FIKS does the job of smoothing the states well, as can be seen by the much smoother output 
 as compared to the UKF data it smooths. It also entirely fixes the initial convergence period 
 the UKF has where it is initialized with a large error and small covariance, so takes a long time to converge
 to the true states. As shown in the fuel remaining plot, the FIKS tracks the true value from time zero with much
 less oscillations. However, it appears as though this is done at the cost of much increased smoothed covariance
 estimates. The FIKS has a variances of over 100 for most of the fuel remaining estimate, while the UKF
 variances remain under 50 for the same data. So it appears that the FIKS, while doing a good job of 
 smoothing the data and keeping the error similar (for this dataset, at least), decreases the confidence in the 
 accuracy of that data as estimated by the filter. '''