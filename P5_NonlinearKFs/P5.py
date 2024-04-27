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
yk = csv_data[:, 1]    # range measurement
hk = csv_data[:, 2]    # true altitude
sk = csv_data[:, 3]    # true vertical velocity
Cbk = csv_data[:, 4]    # true Ballistic Coefficient

# define model parameters
dt = 0.5 # seconds
rho0 = 0.0765 # lb/ft3
g0 = 32.2 # ft/s2
h_rho = 30000.0 # ft
R_E = 20902260.0 #ft
d = 100000.0 # ft
Q = np.diag([10.0**2, 10.0**2, .05**2])
R = 100.0**2 # ft2
x0 = np.array([400000.0, -2000.0, 20.0]) # ft, ft/s, lb/ft2
P0 = np.diag([100.0**2, 10.0**2, 1.0**2]) # ft2, ft2/s2, 

# define state space model for fuel system:
# state vector: [altitude, verticalSpeed, ballistic coefficient]
def propogate_state(x_hat, u=0, w=np.zeros(3)):
    f0 = lambda x_hat: x_hat[0] + dt*x_hat[1] + w[0]
    f1 = lambda x_hat: x_hat[1] + dt*(rho0*x_hat[1]**2/2/x_hat[2]*np.exp(-x_hat[0]/h_rho) - g0*(R_E/(R_E + x_hat[0]))**2) + w[1]
    f2 = lambda x_hat: x_hat[2] + w[2]
    next_state = np.array([f0(x_hat), f1(x_hat), f2(x_hat)])
    return next_state

# test the propogate function
if False:
    xtest = np.zeros((len(x0), len(time)))
    xtest[:,0] = x0
    for i in range(len(time)-1):
        xtest[:,i+1] = propogate_state(xtest[:,i])
    fig, ax = plt.subplots()
    ax.plot(time, xtest[2,:], c='red', label='model')
    ax.plot(time, Cbk, c='black', label='truth')
    ax.set(xlabel = 't, s', ylabel = 'Altitude, ft',
          title = 'Estimated Altitude vs time, ')
    ax.legend()
    plt.grid(True)

def gen_state_jacobian(x_hat):
    
    F = np.array([[1, dt, 0],
                  [
                      dt*(-1/h_rho*rho0*x_hat[1]**2/2/x_hat[2]*np.exp(-x_hat[0]/h_rho) + g0*2*R_E**2/((R_E + x_hat[0])**3)), 
                      1 + dt*rho0*x_hat[1]/x_hat[2]*np.exp(-x_hat[0]/h_rho),
                      -dt*rho0*x_hat[1]**2/(2*x_hat[2]**2)*np.exp(-x_hat[0]/h_rho)
                   ],
                  [0,0,1]
        ])
    # don't want to delete this because it can show very interesting behavior:
    # if x_hat is integers, the **3 operation goes very wrong, but isn't replicable in terminal
    '''print(x_hat[0])
    print( (R_E + x_hat[0]) )
    print( ((R_E + x_hat[0])**3) )'''
    return F

# this shows strange integer vs float behavior not replicable in terminal:
'''F1 = gen_state_jacobian(x0)
print(F1)
print('')

xtest = np.zeros((len(x0), len(time)))
xtest[:,0] = x0
F2 = gen_state_jacobian(xtest[:,0])
print(F2)'''

# test propogating just from function
if False:
    xtest = np.zeros((len(x0), len(time)))
    xtest[:,0] = x0
    for i in range(len(time)-1):
        F = gen_state_jacobian(xtest[:,i])
        xtest[:,i+1] = F@xtest[:,i]
    fig, ax = plt.subplots()
    ax.plot(time, xtest[0,:], c='red', label='model')
    ax.plot(time, hk, c='black', label='truth')
    ax.set(xlabel = 't, s', ylabel = 'Altitude, ft',
          title = 'Estimated Altitude vs time, ')
    ax.legend()
    plt.grid(True)
    plt.ylim(0, 500000)
    
    xtest = np.zeros((len(x0), len(time)))
    xtest[:,0] = x0
    for i in range(len(time)-1):
        F = gen_state_jacobian(xtest[:,i])
        xtest[:,i+1] = F@xtest[:,i]
    fig, ax = plt.subplots()
    ax.plot(time, xtest[1,:], c='red', label='model')
    ax.plot(time, sk, c='black', label='truth')
    ax.set(xlabel = 't, s', ylabel = 'Vspeed, ft',
          title = 'vertical speed vs time, ')
    ax.legend()
    plt.grid(True)
    plt.ylim(0, -4000)

# define measurement model for system
def measure(x_hat, v=0):
    return np.sqrt(d**2 + x_hat[0]**2) + v

def gen_measure_jacobian(x_hat):
    return np.array( [x_hat[0]/np.sqrt(d**2 + x_hat[0]**2), 0, 0] )

def gen_process_noise_jacobian(x_hat):
    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
        ])

def gen_measure_noise_jacobian(x_hat):
    return np.array([1])


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
filters_log['EKF']['gain'] = np.zeros((time.size-1, 3))
filters_log['UKF']['gain'] = np.zeros((time.size-1, 3))

# define function to run filters and log results for data in csv
def runFilter(filter_object, filter_log, log_gain = False):
    for i in range(0,time.size-1):
        # could make faster by not looking up data in log, but this more readable
        x_apriori, P_apriori = filter_object.predict(filter_log['x'][i,:], filter_log['P'][i,:,:])
        x_posteriori, P_posteriori = filter_object.correct(yk[i+1], x_apriori, P_apriori)
        filter_log['x'][i+1,:] = x_posteriori
        filter_log['P'][i+1,:,:] = P_posteriori
        if log_gain:
            filter_log['gain'][i, :] = filter_object.K

# plotting function
def plot(filter_log, filter_string):
    # Plot the posterior state estimates versus the true states for the fuel remaining.
    fig, ax = plt.subplots()
    ax.plot(time, filter_log['x'][:,0], c='red', label='estimated', lw=2)
    ax.plot(time, hk, c='black', label='truth')
    ax.set(xlabel = 't, s', ylabel = 'Altitude, ft',
          title = 'Estimated Altitude vs time, ' + filter_string)
    ax.legend()
    plt.grid(True)
    
    # Plot the posterior state estimates versus the true states for the flow meter bias.
    fig, ax = plt.subplots()
    ax.plot(time, filter_log['x'][:,1], c='red', label='estimated')
    ax.plot(time, sk, c='black', label='truth')
    ax.set(xlabel = 't, s', ylabel = 'Vertical speed, ft/s',
          title = 'Estimated Vertical Speed vs time, ' + filter_string)
    ax.legend()
    plt.grid(True)
    
    # Plot the posterior state estimates versus the true states for the flow meter bias.
    fig, ax = plt.subplots()
    ax.plot(time, filter_log['x'][:,2], c='red', label='estimated')
    ax.plot(time, Cbk, c='black', label='truth')
    ax.set(xlabel = 't, s', ylabel = 'Ballistic Coefficient',
          title = 'Estimated Ballistic Coefficient vs time, ' + filter_string)
    ax.legend()
    plt.grid(True)
    
    # Plot the posterior state estimate errors and the ±2𝜎 from the diagonals of the steady-state
    # posterior covariance for the fuel remaining.
    fig, ax = plt.subplots()
    ax.plot(time, filter_log['x'][:,0]-hk, c='red', label=filter_string)
    ax.fill_between(time, -2*np.sqrt(filter_log['P'][:,0,0]), 2*np.sqrt(filter_log['P'][:,0,0]), color='purple', alpha=0.5, label='2-sigma covariance bound')
    ax.set(xlabel = 't, s', ylabel = 'Altitude',
          title = 'Posterior State Estimate Errors and estimated 2-sigma bounds vs time, ' + filter_string)
    ax.legend()
    plt.grid(True)
    
    # Plot the posterior state estimate errors and the ±2𝜎 from the diagonals of the steady-state
    # posterior covariance for the flow meter bias.
    fig, ax = plt.subplots()
    ax.plot(time, filter_log['x'][:,1]-sk, c='red', label=filter_string)
    ax.fill_between(time, -2*np.sqrt(np.abs(filter_log['P'][:,1,1])), 2*np.sqrt(np.abs(filter_log['P'][:,1,1])), color='purple', alpha=0.5, label='2-sigma covariance bound')
    ax.set(xlabel = 't, s', ylabel = 'Vertical Speed',
          title = 'Posterior State Estimate Errors and estimated 2-sigma bounds vs time, ' + filter_string)
    ax.legend()
    plt.grid(True)
    
    # Plot the posterior state estimate errors and the ±2𝜎 from the diagonals of the steady-state
    # posterior covariance for the flow meter bias.
    fig, ax = plt.subplots()
    ax.plot(time, filter_log['x'][:,2]-Cbk, c='red', label=filter_string)
    ax.fill_between(time, -2*np.sqrt(np.abs(filter_log['P'][:,2,2])), 2*np.sqrt(np.abs(filter_log['P'][:,2,2])), color='purple', alpha=0.5, label='2-sigma covariance bound')
    ax.set(xlabel = 't, s', ylabel = 'Ballistic Coefficient',
          title = 'Posterior State Estimate Errors and estimated 2-sigma bounds vs time, ' + filter_string)
    ax.legend()
    plt.grid(True)
    

#%% standard discrete-time Kalman Filter (EKF)

EKF = P5_classes.EKF(propogate_state, measure, gen_state_jacobian, gen_measure_jacobian, gen_process_noise_jacobian, gen_measure_noise_jacobian, Q, R)
runFilter(EKF, filters_log['EKF'], True)
plot(filters_log['EKF'], 'EKF')


#%% Unscented Kalman Filter (UKF)
UKF = P5_classes.UKF(propogate_state, measure, Q, R, alpha=0.1)
runFilter(UKF, filters_log['UKF'], False)
plot(filters_log['UKF'], 'UKF')

# plot kalman gains vs UKF gains
'''
fig, ax = plt.subplots()
ax.plot(time[1:], filters_log['EKF']['gain'][:,0], c='red', label='EKF fuel remaining gain')
ax.plot(time[1:], filters_log['EKF']['gain'][:,1], c='black', label='EKF flowmeter bias gain')
ax.plot(time[1:], filters_log['UKF']['gain'][:,0], c='blue', label='UKF fuel remaining gain')
ax.plot(time[1:], filters_log['UKF']['gain'][:,1], c='green', label='UKF flowmeter bias gain')
ax.set(xlabel = 't, s', ylabel = 'Kalman gains, cm^3',
      title = 'UKF vs EKF Kalman gains')
ax.legend()
plt.grid(True)
'''

#%% bootstrap particle Kalman Filter (CI-EKF)

'''
BSKF = P5_classes.BSKF(F, G, Q, H, R)
runFilter(BSKF, filters_log['BSKF'])
plot(filters_log['BSKF'], 'BSKF')
'''

'''# fixed-interval Kalman Smoother (FI_KS)
FIKS = P4_classes.FIKS(F, G, Q, H, R)
filters_log['FIKS'] = copy.deepcopy(filters_log['UKF'])
for i in range(time.size-2,-1,-1):
    # could make faster by not looking up data in log, but this more readable
    #x_plus1_posteriori, x_posteriori, u, P_plus1_posteriori, P_posteriori)
    x_smoothed, P_smoothed = FIKS.smooth(filters_log['FIKS']['x'][i+1,:], filters_log['UKF']['x'][i,:], uk[i], filters_log['FIKS']['P'][i+1,:,:], filters_log['UKF']['P'][i,:,:])
    filters_log['FIKS']['x'][i,:] = x_smoothed
    filters_log['FIKS']['P'][i,:,:] = P_smoothed
plot(filters_log['FIKS'], 'FIKS')'''

# commentary
'''
A
'''