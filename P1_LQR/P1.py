"""
AEM566 - Optimal Estimation and Control
Project 1:
    
    Linear Quadratic Regulator Design for Rendezvous and Proximity 
                Opoerations of two simulated spacecraft 


Contents:
    - initialize values & form state space
    - determine stability and controllability of system
    - for different Q and R matrices:
        > finite horizon, continous LQR
        > infinite-horizon, continous LQR
        > finite-horizon, discrete LQR
        > infinite-horizon discrete LQR
    - commentary 
    
All calculations done in hill frame of target spacecraft. 

Written by Joey Westermeyer 2024

"""

import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt

# initialize constants
mu = 3.986004418E14 # earth, m3/s2
rt = 6783000 # target spacecraft altitude (assumed constant), m
x0 = np.array([1000,1000,1000, 0, 0, 0]) # initial chaser state
nt = np.sqrt(mu/rt**3)

# continous time state space. Dynamics use CW-equations (Hill equations) in target's hill frame
A = np.zeros((6,6))
A[3:6,3:6] = np.eye(3)
A[3,0] = 3*nt**2
A[5,2] = -nt**2
A[4,3] = -2*nt
A[3,4] = 2*nt

B = np.block([[np.zeros((3,3))], [np.eye(3)]])



# Compute the eigenvalues of the state matrix for the continuous-time LTI 
# system and comment on the stability of the system

# Compute the controllability matrix for the continuous-time LTI system
