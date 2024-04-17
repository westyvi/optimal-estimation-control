#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 13:30:44 2024

@author: westyvi
"""

import numpy as np
import scipy
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
    
    @abstractmethod
    def measurement(self, x_hat):
        # returns y_hat
        pass
    
    @abstractmethod
    def update_measurement_matrices(self, x_hat, p_hat):
        # update R, H, M(measurement noise gain matrix) self matrices
        pass
    
    # overwrite this for non-standard kalman gain calculations (SS-KF)
    def calculateKalmanGain(self, x_hat, p_hat):
        
        self.K = p_hat @ self.H.T / (self.H @ p_hat @ self.H.T + self.R)
        
        S = self.R + self.H @ (p_hat @ self.H.T)
        self.K = np.dot(p_hat @ self.H.T, (1.0/S))
        #S = self.R + self.H @ (p_hat @ self.H.T)
        #self.K = np.dot(p_hat @ self.H.T, (1.0/S))
        
        '''
        self.update_measurement_matrices(x_hat, p_hat)
        
        # innovation covariance noise covariance matrix S
        if np.isscalar(self.M):
            self.S = self.H @ (p_hat @ self.H.T) + self.M * self.R * self.M
        else:
            self.S = self.H @ p_hat @ self.H.T + self.M @ self.R @ self.M.T
        
        # enforece symmetry 
        #self.S = 0.5*(self.S + self.S.T)
        
        # enforce positive semidefinite
        #self.S = la.sqrtm(self.S.T @ self.S)
        #_ = np.linalg.pinv(self.S)
        #self.S = np.linalg.pinv(_)
        
        # regularize matrix to prevent underflow
        #epsilon = 1E-5
        #self.S = self.S + np.eye(self.S.shape[0])*epsilon
       
        # Kalman gain matrix
        if np.isscalar(self.S):
            self.K = (p_hat @ self.H.T) / (self.S)
        else:
            self.K = p_hat @ self.H.T @ np.linalg.inv(self.S)
        pass'''
    
    def predict(self, x_hat, p_hat, dt, u=0):
       
        self.updatePredictMatrices(x_hat) # has side effect of updating self matrices
        
        # update P_hat to apriori covariance estimate P_k|k-1
        p_hat = self.F @ p_hat @ self.F.T + self.L @ self.Q @ self.L.T
        p_hat = 0.5*(p_hat + p_hat.T) # enforce symmetry
        #epsilon = 1E-07
        #p_hat = p_hat + np.eye(p_hat.shape[0])*epsilon
        
        # update predicted apriori state with nonlinear propogation equation
        x_hat = self.state_propogation(x_hat, u)
        
        return x_hat, p_hat
        
    def correct(self, y_measured, x_hat, p_hat):
        self.calculateKalmanGain(x_hat, p_hat)
        
        # compare expected measurement to sensor measurement
        y_hat = self.measurement(x_hat)
        innovation = y_measured - y_hat
        '''
        # posteriori mean state estimate (from apriori state estimate)
        if np.isscalar(self.K) or np.isscalar(innovation):
            x_hat = x_hat + self.K * innovation
        else:
            x_hat = x_hat + self.K @ innovation
        
        # iterate here to get better measurement matrices if IEKF

        # Joseph form aposteriori covariance update 
        if np.isscalar(self.K):
            A = np.eye(p_hat.shape[0]) - self.K * self.H
        else:
            A = np.eye(p_hat.shape[0]) - self.K @ self.H
            
        if np.isscalar(self.K):
            # FIXME need four cases not two
            p_hat = A @ p_hat @ A.T + self.K * self.R * self.K.T
        else:
            if np.isscalar(self.R): #
                p_hat = A @ p_hat @ A.T + self.R* self.K @ self.K.T
            else:
                p_hat = A @ p_hat @ A.T + self.K @ self.R @ self.K.T
                '''
        
        x_hat = x_hat + self.K * innovation
        A = np.eye(p_hat.shape[0]) - np.outer(self.K,self.H)
        p_hat = A @ p_hat @ A.T + self.R* np.outer(self.K, self.K.T)
        
        return x_hat, p_hat
    
class KF(BaseKF):
    # implements linear, discrete time standard kalman filter
    # if no control input, define G=0
    def __init__(self, F, G, Q, H, R, L=0,M=0):
        self.F = F # state transition matrix
        self.G = G # control matrix
        self.Q = Q # process noise
        self.H = H # measurement
        self.R = R # measurement noise
        if L == 0:
            self.L = np.eye(Q.shape[1])
        if M == 0:
            if np.isscalar(R):
                self.M = 1
            else:
                self.M = np.eye(R.shape[1])

    
    def updatePredictMatrices(self, x_hat):
        # no update needed for linear KF
        pass
   
    def state_propogation(self, x_hat, u=0):
        # does not handle if state, F, or G are scalar
        if np.isscalar(u):
            return self.F @ x_hat + self.G * u
        return self.F @ x_hat + self.G @ u
        
    def measurement(self, x_hat):
        return self.H @ x_hat
    
    def update_measurement_matrices(self, x_hat, p_hat):
        # no update needed for linear KF
        pass
    
class SSKF(KF):
    def calculateKalmanGain(self, x_hat, p_hat):
        self.update_measurement_matrices(x_hat, p_hat)
        
        # solve for infinite horizon steady state cost P, then SS kalman gain K
        P = scipy.linalg.solve_discrete_are(self.F.T,np.array([[self.H[0]],[self.H[1]]]),self.Q,self.R) 
        self.K = 1/(self.H.T @ P @ self.H + self.R) * self.H.T @ P @ self.F.T
        
        # try number 2 (from paper)
        D = self.H @ P @ self.H.T + self.R
        G2 = P @ self.H.T /(D) # different from control matrix self.G
        self.K = self.F @ G2
    
class CIKF(KF):
    def correct(self, y_measured, x_hat, p_hat):
        # compare expected measurement to sensor measurement
        y_hat = self.measurement(x_hat)
        innovation = y_measured - y_hat
        
        # solve covariance intersection optimization problem for wopt (w_optimal)
        P_inv = np.linalg.inv(p_hat)
        if np.isscalar(self.R):
            R_inv =1/self.R
            objective = lambda omega: np.trace(np.linalg.inv(omega*P_inv + (1 - omega)*self.H.T * R_inv @ self.H))
        else:
            R_inv = np.linalg.inv(self.R)
            objective = lambda omega: np.trace(np.linalg.inv(omega*P_inv + (1 - omega)*self.H.T @ R_inv @ self.H))
           
        res = scipy.optimize.minimize_scalar(objective, bounds=(0,0.95), method='bounded')
        wopt = res.x
        
        # calculate aposteriori covariance and mean
        if np.isscalar(self.R):
            p_hat_posteriori = np.linalg.inv(wopt*P_inv + (1-wopt)*self.H.T * R_inv @ self.H)
            self.K = (1-wopt)*p_hat_posteriori @ self.H.T * R_inv
        else:
            p_hat = np.linalg.inv(wopt*P_inv + (1-wopt)*self.H.T@ R_inv @ self.H)
            self.K = (1-wopt)*p_hat @ self.H.T @ R_inv
            
        # p_hat = 0.5*p_hat + 0.5 * p_hat.T # enforce symmetry 
        if np.isscalar(innovation):
            x_hat = x_hat + self.K * innovation
        else:
            x_hat = x_hat + self.K @ innovation
        
        return x_hat, p_hat_posteriori
    
class FIKS(KF):
    # this assumes u is a scalar
    def smooth(self, x_plus1_posteriori, x_posteriori, u, P_plus1_posteriori, P_posteriori):
        innovation = x_plus1_posteriori - self.F @ x_posteriori - self.G * u
        Ks = P_posteriori @ self.F.T @ np.linalg.inv(self.F @ P_posteriori @ self.F.T + self.Q)
        x_smoothed = x_posteriori + Ks @ innovation
        P_smoothed = P_posteriori + Ks @ (P_plus1_posteriori - self.F @ P_posteriori @ self.F.T - self.Q) @ Ks.T
        return x_smoothed, P_smoothed