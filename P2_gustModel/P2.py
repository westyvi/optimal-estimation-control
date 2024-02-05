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
from scipy.integrate import solve_ivp as ode45
import scipy

# initialize constants
dt = 0.01 # s

# rms_gust x,y, and z are equivalent for this problem
rms_gust_light = 5 # ft/s
rms_gust_moderate = 10 # ft/s
rms_gust_severeS = 20 # ft/s
vinf = 824 # ft/s
h = 20E3 # ft

# Lu, Lv, Lw are equivalent to 1750 since h>1750, per the Dryden gust model 
Lu = 1750 
Lv = 1750
Lw = 1750

print(rms_gust_light*np.sqrt(2*vinf/(np.pi*Lu)))
# implement discretized version of Dryden gust model 


