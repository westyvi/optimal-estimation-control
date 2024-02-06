"""
AEM566 - Optimal Estimation and Control
Project 2:
    
    Wind Simulator Design using the Dryden stochastic gust model

Contents:
    - initialize values & form state space
    - implement discretized version of gust model 
    - Simulate gust model for multiple cases
    - Plot results
    
Commentary contained in project folder
    
Written by Joey Westermeyer 2024

notes:
    
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp as ode45 # sweet home matlab
import scipy

# initialize constants
dt = 0.01 # s
simTime = 600 # s
numpts = int(simTime/dt) + 1 # sim includes start (0) AND end point

# rms_gust x,y, and z are equivalent for this problem, so can condense to one rms value for each case
rms_gust_light = 5 # ft/s
rms_gust_moderate = 10 # ft/s
rms_gust_severe = 20 # ft/s
vinf = 824 # ft/s
h = 20E3 # ft
noise_variance = 1/dt
noise_stddev = noise_variance**0.5

# Lu, Lv, Lw are equivalent to 1750 since h>1750, per the Dryden gust model 
Lu = 1750 
Lv = 1750
Lw = 1750

# make container of simulation cases to make looping over cases easier
sim_runs = {'light': {'rms_gust' : rms_gust_light, 'color': 'g'},
            'moderate': {'rms_gust' : rms_gust_moderate, 'color': 'b'},
            'severe': {'rms_gust': rms_gust_severe, 'color' : 'r'}}

# implement discretized version of Dryden gust model 
def generate_discrete_gust_state_space(rms_gust, vinf, Lu, Lv, Lw, dt):
    # state is [udot, vdot, vdot1, wdot, wdot1].T
    # (vdot1 and wdot1 serve as 'memory' variables to make this a second order markov process)
    
    # continuous time gust model
    A = np.diag([-vinf/Lu, -vinf/Lv, -vinf/Lv, -vinf/Lw, -vinf/Lw])
    A[1,2] = rms_gust*(1-np.sqrt(3))*(vinf/Lv)**(3/2)
    A[3,4] = rms_gust*(1-np.sqrt(3))*(vinf/Lw)**(3/2)
    B = np.array([rms_gust*(2*vinf/np.pi/Lu)**.5, rms_gust*(3*vinf/np.pi/Lv)**.5, 1, rms_gust*(3*vinf/np.pi/Lw)**.5, 1])

    # approximate with different methods (for comparison if interested)
    F_forwardEuler = np.eye(A.shape[0]) + A*dt
    F_backEuler = np.linalg.inv(np.eye(A.shape[0]) - A*dt)
    F_bilinear = (np.eye(A.shape[0]) + 0.5*A*dt) @ np.linalg.inv((np.eye(A.shape[0]) + 0.5*A*dt))
    G_forwardEuler = B*dt

    # solve exactly
    F = scipy.linalg.expm(A*dt)
    G = np.linalg.inv(A) @ (F - np.eye(F.shape[0])) @ B

    return F,G

# Populate sim cases with corresponding state spaces for Dryden gust model 
for sim_case in sim_runs.values():
    sim_case['F'], sim_case['G'] = generate_discrete_gust_state_space(sim_case['rms_gust'], vinf, Lu, Lv, Lw, dt)

# simulate Dryden gust model, output uvw gust values
def simulate_gust_model(F, G, numpts, noise_stddev):
    uvw_gust = np.zeros((3, numpts))
    x = np.zeros(5)
    rng = np.random.default_rng()
    
    for i in range(numpts):
        uvw_gust[:,i] = [x[index] for index in [0, 1, 3]] # select u,v,w from x
        noise = rng.normal(0,noise_stddev)
        x = F @ x + G * noise
        
    return uvw_gust

# simulate gust model for each case and output to sim case
for sim_case in sim_runs.values():
    sim_case['uvw_gust'] = simulate_gust_model(sim_case['F'], sim_case['G'], numpts, noise_stddev)


# %%plot results
t = np.linspace(0, simTime, numpts)
fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()
figs = [fig1, fig2, fig3]
axes = [ax1, ax2, ax3]
xyz_string_list = ['x', 'y', 'z']

# loop through each simulation case 
for sim_case in sim_runs:
    # loop through each data set (vx, vy, and vz)
    for i, ax in enumerate(axes):
        ax.plot(t, sim_runs[sim_case]['uvw_gust'][i,:], sim_runs[sim_case]['color'], label=sim_case)
        ax.set(xlabel = 't, s', ylabel = 'v, ft/s',
              title = xyz_string_list[i] + ' gust vs time')
        ax.legend()
        ax.grid(True)
   

