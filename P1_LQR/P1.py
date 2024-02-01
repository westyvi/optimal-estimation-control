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
    
All calculations done in Hill frame of target spacecraft. 

Written by Joey Westermeyer 2024

notes:
    Want to make a better plotting function. More DRY. 
        Potentially convert it to accept a dict instead of simple namespace for more general solutions
    Want to examine using e^At as the discretized matrix instead of the given derivation
    Want to examine how K changes over time for finite vs infinite horizons
    Want to examine response to sensor & process noise
    Want to examine eLQR with nonlinear dynamics equation
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp as ode45
import scipy
from types import SimpleNamespace

# initialize constants
mu = 3.986004418E14 # earth, m3/s2
rt = 6783000 # target spacecraft altitude (assumed constant), m
x0 = np.array([1000,1000,1000, 0, 0, 0]) # initial chaser state
nt = np.sqrt(mu/rt**3)

# continous time state space. Dynamics use CW-equations (Hill equations) in target's hill frame
A = np.zeros((6,6))
A[0:3,3:6] = np.eye(3)
A[3,0] = 3*nt**2
A[5,2] = -nt**2
A[4,3] = -2*nt
A[3,4] = 2*nt

B = np.block([[np.zeros((3,3))], [np.eye(3)]])

# discrete time state space. Derived from continous time state space CW equations
dt = 1
F = np.zeros((6,6))
F[0:3,0:3] = [[4-3*np.cos(nt*dt), 0, 0],
              [6*(np.sin(nt*dt) - nt*dt), 1, 0],
              [0, 0, np.cos(nt*dt)]
              ]
F[3:6,0:3] = [[3*nt*np.sin(nt*dt), 0, 0],
              [-6*nt*(1-np.cos(nt*dt)), 0, 0],
              [0, 0, -nt*np.sin(nt*dt)]
              ]
F[0:3,3:6] = [[nt**-1*np.sin(nt*dt), 2*nt**-1*(1-np.cos(nt*dt)), 0],
              [-2*nt**-1*(1-np.cos(nt*dt)), nt**-1*(4*np.sin(nt*dt) - 3*nt*dt), 0],
              [0, 0, nt**-1*np.sin(nt*dt)]
              ]
F[3:6,3:6] = [[np.cos(nt*dt), 2*np.sin(nt*dt), 0],
              [-2*np.sin(dt*nt), 4*np.cos(nt*dt) - 3, 0],
              [0, 0, np.cos(nt*dt)]
              ]

G = np.zeros((6,3))
G[0:2,0:2] = [[nt**-2*(1-np.cos(nt*dt)), 2*nt**-2*(nt*dt - np.sin(nt*dt))],
              [-2*nt**-2*(nt*dt - np.sin(nt*dt)), 4*nt**-2*(1-np.cos(nt*dt)) - dt**2*3/2]
              ]
G[2,2] = nt**-2*(1-np.cos(nt*dt))
G[3:5,0:2] = [[nt**-1*np.sin(nt*dt), 2*nt**-1*(1-np.cos(nt*dt))],
              [-2*nt**-1*(1-np.cos(nt*dt)), 4*nt**-1*np.sin(dt*dt) - 3*dt]
              ]
G[5,2] = nt**-1*np.sin(nt*dt)

# Compute the eigenvalues of the state matrix for the continuous-time LTI 
# system and comment on the stability of the system
cont_eigs, cont_eigvects = np.linalg.eig(A)
print(cont_eigs)
print()
print('All eigenvalues are purely imaginary, which for continous time means '+\
    'the states will oscillate, but will not grow unbounded or settle to a ' +\
    'stable value. In other words, the system described by the linearized '+\
    'dynamics of the state matrix A is marginally stable.\n')

# Compute the controllability matrix for the continuous-time LTI system
controllability_matrix = np.block([B, A @ B, A@A @ B, A@A@A @ B, A@A@A@A @ B, A@A@A@A@A @ B])
control_rank = np.linalg.matrix_rank(controllability_matrix)
print(control_rank)
print('^Controllability matrix rank. The controllability matrix has full rank, so the system is controllable')
print()

# stability & controllability of discretized state space
disc_eigs, disc_eigvects = np.linalg.eig(F)
#print(disc_eigs)
#print('^discrete eigenvalues')
print(np.abs(disc_eigs))
print('^discrete eigenvalue absolute values. All magnitudes are equal to '+\
      ' 1, which signifies that the system is marginally stable')


# setup cases 1, 2, and 3 for Q and R matrix design
AssignmentGaveUsTheWrongScaleFactor = 1E3
Q1 = np.eye(6)
R1 = np.eye(3)*AssignmentGaveUsTheWrongScaleFactor
Q2 = np.eye(6)
R2 = 100*np.eye(3)* AssignmentGaveUsTheWrongScaleFactor
Q3 = np.eye(6)
R3 = 10000*np.eye(3)* AssignmentGaveUsTheWrongScaleFactor
Qs = [Q1, Q2, Q3]
Rs = [R1, R2, R3]
simTime = 400 # s
numpts = 1500 # number of evaluation points for continuous time, number of time steps for discrete time

    
def plot(sol, designString):
    
    fig, ax = plt.subplots()
    ax.plot(sol[0].y[0,:], sol[0].y[1,:], 'b', label='case 1')
    ax.plot(sol[1].y[0,:], sol[1].y[1,:], 'r', label='case 2')
    ax.plot(sol[2].y[0,:], sol[2].y[1,:], 'g', label='case 3')
    ax.set(xlabel = 'x, m', ylabel = 'y, m',
          title = 'xy trajectory, ' + designString)
    ax.legend()
    plt.grid(True)
    
    fig, ax = plt.subplots()
    ax.plot(sol[0].t, sol[0].y[2,:], 'b', label='case 1')
    ax.plot(sol[1].t, sol[1].y[2,:], 'r', label='case 2')
    ax.plot(sol[2].t, sol[2].y[2,:], 'g', label='case 3')
    ax.set(xlabel = 't, s', ylabel = 'z, m',
          title = 'z-time trajectory, ' + designString)
    ax.legend()
    plt.grid(True)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3,1)
    fig.suptitle('control input vs time, ' + designString)
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    
    ax1.plot(sol[0].t, sol[0].u[0,:], 'b', label='case 1')
    ax1.plot(sol[1].t, sol[1].u[0,:], 'r', label='case 2')
    ax1.plot(sol[2].t, sol[2].u[0,:], 'g', label='case 3')
    ax1.set(xlabel = 't, s', ylabel = 'U_x, m/s2')
    
    ax2.plot(sol[0].t, sol[0].u[1,:], 'b', label='case 1')
    ax2.plot(sol[1].t, sol[1].u[1,:], 'r', label='case 2')
    ax2.plot(sol[2].t, sol[2].u[1,:], 'g', label='case 3')
    ax2.set(xlabel = 't, s', ylabel = 'U_y, m/s2')
    ax2.legend()
    
    ax3.plot(sol[0].t, sol[0].u[2,:], 'b', label='case 1')
    ax3.plot(sol[1].t, sol[1].u[2,:], 'r', label='case 2')
    ax3.plot(sol[2].t, sol[2].u[2,:], 'g', label='case 3')
    ax3.set(xlabel = 't, s', ylabel = 'U_z, m/s2')

    
# %% finite-horizon, continous time LQR
def continous_Riccati(t, P):
    P = P.reshape(6,6)
    Pdot = -P @ A - A.T @ P + P @ B @ np.linalg.inv(R) @ B.T @ P - Q
    return Pdot.flatten()
           
Pf = np.zeros((6,6))

sols = []
for i in range(3):
    R = Rs[i]
    Q = Qs[i]
    Ps = ode45(continous_Riccati, [simTime, 0], Pf.flatten(),
                   t_eval=np.linspace (simTime, 0, numpts)) # returns Ps in array of time from t_N -> t_0
    
    # there is definitely a better way to do this with numpy matrix commands
    Ks = np.zeros((3,6,numpts))
    Ksflat = np.zeros((18,numpts))
    for j in range (numpts):
        Ks[:,:,j] = np.linalg.inv(R) @ B.T @ Ps.y[:,numpts-j-1].reshape(6,6)
        Ksflat[:,j] = Ks[:,:,j].flatten()
    
    # simulate system
    interpK = scipy.interpolate.interp1d(Ps.t, Ksflat)
    sys_CL = lambda t, x: (A - B @ interpK(t).reshape(3,6)) @ x
    
    sol = ode45(sys_CL, [0, simTime], x0, t_eval=np.linspace(0,simTime, numpts))
    
    # recover control history 
    u = np.zeros((3, sol.y.shape[1]))
    for i in range(sol.y.shape[1]):
        u[:,i] = -interpK(sol.t[i]).reshape(3,6) @ sol.y[:,i]
        
    # output simulation and control history
    sol.u = u
    sols.append(sol)
    
plot(sols, 'finite horizon continuous LQR')


# %% infinite-horizon, continous time LQR

sols = []
Ks = []
# loop through cases
for i in range(3):
    
    # solve for cost to go matrix P and optimal gain K
    R = Rs[i]
    Q = Qs[i]
    P = scipy.linalg.solve_continuous_are(A,B,Q,R)
    K = np.linalg.inv(R) @ B.T @ P
    Ks.append(K)
    
    # simulate system
    sys_CL = lambda t, x: (A - B @ K) @ x
    sol = ode45(sys_CL, [0, simTime], x0, t_eval=np.linspace(0,simTime, numpts))
    
    # recover control history 
    u = np.zeros((K.shape[0], sol.y.shape[1]))
    for i in range(sol.y.shape[1]):
        u[:,i] = -K @ sol.y[:,i]
        
    # output simulation and control history
    sol.u = u
    sols.append(sol)

plot(sols, 'infinite horizon continuous LQR')

# %% finite-horizon, discrete time LQR
 
sols = []
dt = simTime/numpts
for i in range(3):
    Ps = np.zeros((6,6,numpts))
    K_history = np.zeros((6,6,numpts))
    u_history = np.zeros((3,numpts+1)) # last value will be left zero, but need it there to match length with x
    x_history = np.zeros((6,numpts+1))
    x_history[:,0] = x0
    t_history = np.zeros(numpts+1)
    
    R = Rs[i]
    Q = Qs[i]
    
    for j in range (numpts):
        P = Ps[:,:,numpts-j-1]
        Ps[:,:,numpts-j-2] = F.T @ P @ F + Q - (F.T @ P @ G) @ np.linalg.inv(G.T @ P @ G + R) @ (G.T @ P @ F)
        # Ps runs from k=1 to k=N (so this indexing is shifted off the 'physical'/mathematical indexing by one)
    
    # solve for K, simulate system, log output
    for j in range (numpts):
        t_history[j+1] = t_history[j] + dt
        P = Ps[:,:,j] # want P[k+1] to calculate K[k], and Ps are 1 index ahead of Ks, so use same index j
        K  = np.linalg.inv(G.T @ P @ G + R) @ G.T @ P @ F
        u_history[:,j] = -K @ x_history[:,j]
        x_history[:,j+1] = (F - G @ K) @ x_history[:,j]
    
    # convert solution to format output by ode solver for homogenity
    soldict = {'y':x_history, 'u': u_history, 't':t_history}
    sol = SimpleNamespace(**soldict)
    sols.append(sol)
    
plot(sols, 'finite horizon discrete LQR')

# %% infinite-horizon, continous time LQR

sols = []
# loop through cases
for i in range(3):
    # initialize results matrices
    u_history = np.zeros((3,numpts+1)) # last value will be left zero, but need it there to match length with x
    x_history = np.zeros((6,numpts+1))
    x_history[:,0] = x0
    t_history = np.zeros(numpts+1)
    
    # solve for cost to go matrix P and optimal gain K
    R = Rs[i]
    Q = Qs[i]
    P = scipy.linalg.solve_discrete_are(F,G,Q,R)
    K = np.linalg.inv(G.T @ P @ G + R) @ G.T @ P @ F
    
    # solve for K, simulate system, log output
    for j in range (numpts):
        t_history[j+1] = t_history[j] + dt
        u_history[:,j] = -K @ x_history[:,j]
        x_history[:,j+1] = (F - G @ K) @ x_history[:,j]
    
    # convert solution to format output by ode solver for homogenity
    soldict = {'y':x_history, 'u': u_history, 't':t_history}
    sol = SimpleNamespace(**soldict)
    sols.append(sol)
    
plot(sols, 'infinite horizon discrete LQR')

        
    


                              
                                  