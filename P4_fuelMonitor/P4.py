#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AEM566 - Optimal Estimation and Control
Project 4:
    
    Kalman Filtering for fuel estimation 

initial assignment contained in project folder
    
Written by Joey Westermeyer 2024
    
"""

import numpy as np
from matplotlib import pyplot as plt
import csv
import P4_classes
import copy
import scipy

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
Q = np.diag([A_line**2 * dt**2 * 0.1**2, 0.1**2]) # cm^6, cm^2/s^2
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
filters_log['KF'] = copy.deepcopy(template)
filters_log['SSKF'] = copy.deepcopy(template)
filters_log['CIKF'] = copy.deepcopy(template)
filters_log['FIKS'] = copy.deepcopy(template)

# add gain logging for KF, SSKF
filters_log['KF']['gain'] = np.zeros((time.size-1, 2))
filters_log['SSKF']['gain'] = np.zeros((time.size-1, 2))

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
          title = 'Fuel Remaining Errors and estimated 2-sigma bounds vs time, ' + filter_string)
    ax.legend()
    plt.grid(True)
    
    # Plot the posterior state estimate errors and the ±2𝜎 from the diagonals of the steady-state
    # posterior covariance for the flow meter bias.
    fig, ax = plt.subplots()
    ax.plot(time, filter_log['x'][:,1]-bk, c='red', label=filter_string)
    ax.fill_between(time, -2*np.sqrt(np.abs(filter_log['P'][:,1,1])), 2*np.sqrt(np.abs(filter_log['P'][:,1,1])), color='purple', alpha=0.5, label='2-sigma covariance bound')
    ax.set(xlabel = 't, s', ylabel = 'Flowmeter Bias, cm',
          title = 'Flowmeter Bias Errors and estimated 2-sigma bounds vs time, ' + filter_string)
    ax.legend()
    plt.grid(True)
    
# steady-state Kalman Filter (SS_KF)

SSKF = P4_classes.SSKF(F, G, Q, H, R)
runFilter(SSKF, filters_log['SSKF'], True)
plot(filters_log['SSKF'], 'SSKF')

#%% standard discrete-time Kalman Filter (KF)
KF = P4_classes.KF(F, G, Q, H, R)
#runFilter(KF, filters_log['KF'], True)

# calculate SSKF gain for comparison purposes
filter_object = KF
filter_log = filters_log['KF']
Pcost = scipy.linalg.solve_discrete_are(F.T,np.array([[H[0]],[H[1]]]),Q,R) 
Kss1 = 1/(H.T @ Pcost @ H + R) * H.T @ Pcost @ F.T
# try number 2 (from paper)
D = H @ Pcost @ H.T + R
G2 = Pcost @ H.T /(D) # different from control matrix G
Kss2 = F @ G2

for i in range(0,time.size-1):
    # could make faster by not looking up data in log, but this more readable
    x_apriori, P_apriori = KF.predict(filter_log['x'][i,:], filter_log['P'][i,:,:], dt, uk[i])
    x_posteriori, P_posteriori = KF.correct(yk[i+1], x_apriori, P_apriori)
    K = KF.K
    
    '''# raw correct step instead of OOP
    Mtest = KF.H @ x_apriori
    Stest = KF.R + KF.H @ (P_apriori @ KF.H.T)
    Ktest = np.dot(P_apriori @ KF.H.T, (1.0/Stest))
    #Ktest = Kss2 # overwrite to SSKF gain
    Xtest = x_apriori + np.dot(Ktest, (yk[i+1] - Mtest))
    
    PtestEE = ((np.eye(len(Xtest))- Ktest @ KF.H) @ P_apriori) @ ((np.eye(len(Xtest))-Ktest @ KF.H).T) + Ktest @ np.dot(KF.R, Ktest.T)
    Atest = np.eye(P_apriori.shape[0]) - Ktest * H
    Ptest = Atest @ P_apriori @ Atest.T + Ktest * R * Ktest.T
    print(Ptest-PtestEE)'''
    
    filter_log['x'][i+1,:] = x_posteriori
    filter_log['P'][i+1,:,:] = P_posteriori
    filter_log['gain'][i, :] = K

plot(filters_log['KF'], 'KF')


# plot kalman gains vs SSKF gains
fig, ax = plt.subplots()
ax.plot(time[1:], filters_log['KF']['gain'][:,0], c='red', label='KF fuel remaining gain')
ax.plot(time[1:], filters_log['KF']['gain'][:,1], c='black', label='KF flowmeter bias gain')
ax.plot(time[1:], filters_log['SSKF']['gain'][:,0], c='blue', label='SSKF fuel remaining gain')
ax.plot(time[1:], filters_log['SSKF']['gain'][:,1], c='green', label='SSKF flowmeter bias gain')
ax.set(xlabel = 't, s', ylabel = 'Kalman gains, cm^3',
      title = 'SSKF vs KF Kalman gains')
ax.legend()
plt.grid(True)

#%%
# covariance intersection Kalman Filter (CI-KF)
# bound omega <= 0.95 for search stability
CIKF = P4_classes.CIKF(F, G, Q, H, R)
runFilter(CIKF, filters_log['CIKF'])
plot(filters_log['CIKF'], 'CIKF')

# fixed-interval Kalman Smoother (FI_KS)
FIKS = P4_classes.FIKS(F, G, Q, H, R)
filters_log['FIKS'] = copy.deepcopy(filters_log['KF'])
for i in range(time.size-2,-1,-1):
    # could make faster by not looking up data in log, but this more readable
    #x_plus1_posteriori, x_posteriori, u, P_plus1_posteriori, P_posteriori)
    x_smoothed, P_smoothed = FIKS.smooth(filters_log['FIKS']['x'][i+1,:], filters_log['KF']['x'][i,:], uk[i], filters_log['FIKS']['P'][i+1,:,:], filters_log['KF']['P'][i,:,:])
    filters_log['FIKS']['x'][i,:] = x_smoothed
    filters_log['FIKS']['P'][i,:,:] = P_smoothed
plot(filters_log['FIKS'], 'FIKS')

# commentary
'''As can be seen by the plots produced, the kalman filtering framework provides multiple
ways to estimate states via sensor fusion. As implemented, the base kalman filter struggles to 
track the true fuel remaining and flowmeter bias. While it appears as though this is caused 
by too low a process noise matrix, leading to too much trust in the model 
(shown by small covariance and out of bounds errors), the SSKF child class
performs much better with all the same equations save for the kalman gain calculation, which is constant
and derived from the LQE DARE problem for the SSKF. This is further supported by the kalman gain plot, which 
shows that the KF does not converge to the SSKF gains. I've checked the standard kalman filter gain calculation,
tried different expressions, and re-coded the function but haven't found the culprit for this behavior yet. 
However, in a normal scenario the KF should converge to the SSKF and have the advantage of being able to 
calculate the kalman gain adaptively online, leading to a more flexible and adaptive, while very similar, 
 solution as compared to the KF. It can also be seen that the SSKF approximates the covariances 
 of the states well, with the 95% bounds on both fuel remaining and meter bias containing the errors about 
 95 percent of the time. 
 
 Moving on, the CI-KF can be seen to have worse performance, as it 
 struggles to fix steady state fuel remaining value with overly-small state covariances, then suddenly
 blows up in terms of covariance and erratically oscillates around the true states, seemingly 
 only chasing the measured values, discarding the prediction. This leads to very large errors, oscillations in 
 state, and covariances for most of the data window. So while the CI-KF may be useful in 
 sensor fusion of unknown correlation or in different contexts, it appears to be a poor choice for 
 fusing this data.
 
 Finally, the FIKS does the job of smoothing the states well, as can be seen by the much smoother output 
 as compared to the SSKF data it smooths. It also entirely fixes the initial convergence period 
 the SSKF has where it is initialized with a large error and small covariance, so takes a long time to converge
 to the true states. As shown in the fuel remaining plot, the FIKS tracks the true value from time zero with much
 less oscillations. However, it appears as though this is done at the cost of much increased smoothed covariance
 estimates. The FIKS has a variances of over 100 for most of the fuel remaining estimate, while the SSKF
 variances remain under 50 for the same data. So it appears that the FIKS, while doing a good job of 
 smoothing the data and keeping the error similar (for this dataset, at least), decreases the confidence in the 
 accuracy of that data as estimated by the filter. '''