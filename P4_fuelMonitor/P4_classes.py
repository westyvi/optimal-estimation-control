#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 13:30:44 2024

@author: westyvi
"""

import numpy as np
import math
import copy
from abc import ABC, abstractmethod

class BaseKF(ABC):
    # general kalman filter class
    def __init__(self):
        pass
    
    @abstractmethod
    def updatePredictMatrices(self, x_hat):
        # update the F,Q, and L matrices of self object for covariance propogation
        pass
   
    @abstractmethod
    def state_propogation(self, x_hat):
        # propogate the state
        # return x_hat
        pass
    
    @abstractmethod()
    def measurement(self, x_hat):
        # returns y_hat
        pass
    
    @abstractmethod
    def update_measurement_matrices(self, x_hat, p_hat):
        # updates R, H, M(measurement noise gain matrix) self matrices
        pass
    
    
    # (at the current state estimate x_hat) as class properties
    def predict(self, x_hat, p_hat, dt):
       
        self.F, self.Q, self.L = self.updatePredictMatrices()
        
        # update P_hat to apriori covariance estimate P_k|k-1
        p_hat = self.F @ p_hat @ self.F.T + self.L @ self.Q @ self.L.T
        p_hat = 0.5*(p_hat + p_hat.T) # enforce symmetry
        #epsilon = 1E-07
        #p_hat = p_hat + np.eye(p_hat.shape[0])*epsilon
        
        # update predicted apriori state with nonlinear propogation equation
        x_hat = self.state_propogation(x_hat)
        
        return x_hat, p_hat
        
    # overwrite this for non-standard kalman gain calculations (SS-KF)
    def calculateKalmanGain(self, x_hat, p_hat):
        self.R, self.H, self.M = self.update_measurement_matrices(x_hat, p_hat)
        
        # innovation covariance noise covariance matrix S
        self.S = self.H @ p_hat @ self.H.T + self.M @ self.R @ self.M.T
        
        # enforece symmetry 
        self.S = 0.5*(self.S + self.S.T)
        
        # enforce positive semidefinite
        #self.S = la.sqrtm(self.S.T @ self.S)
        #_ = np.linalg.pinv(self.S)
        #self.S = np.linalg.pinv(_)
        
        # regularize matrix to prevent underflow
        #epsilon = 1E-5
        #self.S = self.S + np.eye(self.S.shape[0])*epsilon
       
        # Kalman gain matrix
        self.K = p_hat @ self.H.T @ np.linalg.inv(self.S)
        
        pass
    
    def measurement_correct(self, y_measured, x_hat, p_hat):
        self.calculateKalmanGain(x_hat, p_hat)
        
        # compare expected measurement to sensor measurement
        y_hat = self.measurement(x_hat)
        innovation = y_measured - y_hat
        
        # posteriori mean state estimate (from apriori state estimate)
        x_hat1 = x_hat + self.K @ innovation
        
        # iterate to get better measurement matrices (this makes this an iterative kalman filter: an IEKF )
        self.update_measurement_matrices(x_hat1, p_hat)
        x_hat = x_hat + self.K @ innovation # FIXME this line is redundant?
        
        # Joseph form aposteriori covariance update 
        A = np.eye(p_hat.shape[0]) - self.K @ self.H
        p_hat = A @ p_hat @ A.T + self.K @ self.R @ self.K.T
        
        return x_hat, p_hat
    
class KF(BaseKF):
    def __init__(self, F, G, Q, H, R):
        self.F = F
        self.G = G
        self.Q = Q
        self.H = H
        self.R = R
        pass
    
    def updatePredictMatrices(self, x_hat):
        # no update needed for linear KF
        pass
   
    def state_propogation(self, x_hat, u):
        return self.F @ x_hat + self.G @ u
        
    def measurement(self, x_hat):
        return self.H @ x_hat
    
    def update_measurement_matrices(self, x_hat, p_hat):
        # no update needed for linear KF
        pass
    
    
    