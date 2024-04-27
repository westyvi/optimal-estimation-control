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
        self.update_measurement_matrices(x_hat, p_hat)
        self.K = p_hat @ self.H.T / (self.H @ p_hat @ self.H.T + self.R)
        
        S = self.R + self.H @ (p_hat @ self.H.T)
        self.K = np.dot(p_hat @ self.H.T, (1.0/S))
       
        '''
        
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
    
    def predict(self, x_hat, p_hat, u=0):
       
        self.updatePredictMatrices(x_hat) # has side effect of updating self matrices F and L
        
        # update P_hat to apriori covariance estimate P_k|k-1
        p_hat = self.F @ p_hat @ self.F.T + self.L @ self.Q @ self.L.T
        p_hat = 0.5*(p_hat + p_hat.T) # enforce symmetry
        epsilon = 1E-07
        p_hat = p_hat + np.eye(p_hat.shape[0])*epsilon
        
        # update predicted apriori state with nonlinear propogation equation
        x_hat = self.state_propogation(x_hat, u)
        
        return x_hat, p_hat
        
    def correct(self, y_measured, x_hat, p_hat):
        self.calculateKalmanGain(x_hat, p_hat)
        
        # compare expected measurement to sensor measurement
        y_hat = self.measurement(x_hat)
        innovation = y_measured - y_hat
        
        # posteriori mean state estimate (from apriori state estimate)
        if np.isscalar(self.K) or np.isscalar(innovation):
            x_hat = x_hat + self.K * innovation
        else:
            x_hat = x_hat + self.K @ innovation
        
        # iterate here to get better measurement matrices if IEKF

# FIXME this is very bad. There are four cases for the different 1 vs >1 ndim options on x and y, and
# this does not cover them all. It is also only debugged for ndim(x)>1 and ndim(y)=1
        # Joseph form aposteriori covariance update 
        if np.isscalar(self.K): # both x and y are one dimension
            A = np.eye(p_hat.shape[0]) - self.K * self.H
            p_hat = A * p_hat * A + self.K * self.R * self.K
        elif self.H.ndim == 1: # FIXME this only handles case where ndim(y)=1 and ndim(x)>1, not the other way around (that would need inner product)
            A = np.eye(p_hat.shape[0]) - np.outer(self.K, self.H)
            p_hat = A @ p_hat @ A.T + self.R* np.outer(self.K, self.K)
        elif False: # this needs to be for ndim(x)=1 and ndim(y)>1
            A = np.eye(p_hat.shape[0]) - np.inner(self.K, self.H)
            p_hat = A * p_hat * A.T + np.inner((self.K @ self.R), self.K)
        else: # x and y have ndim>1
            A = np.eye(p_hat.shape[0]) - self.K @ self.H
            p_hat = A @ p_hat @ A.T + self.K @ self.R @ self.K.T
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
        
        # solf for infinite horizon steady state cost P, then SS kalman gain K
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

class EKF(BaseKF):
    def __init__(self, f, y, fx_jac, y_jac, process_noise_jac, measure_noise_jac, Q, R, fu_jac=0):
        self.propogate_function = f
        self.measure_function = y
        self.state_jac_function = fx_jac
        self.control_jac_function = fu_jac
        self.measure_jac_function = y_jac
        self.process_noise_jac_function = process_noise_jac
        self.measure_noise_jac_function = measure_noise_jac
        self.Q = Q
        self.R = R
        
    def updatePredictMatrices(self, x_hat):
        self.F = self.state_jac_function(x_hat)
        self.L = self.process_noise_jac_function(x_hat)
   
    def state_propogation(self, x_hat, u=0):
        return self.propogate_function(x_hat,u)
        
    def measurement(self, x_hat):
        return self.measure_function(x_hat)
    
    def update_measurement_matrices(self, x_hat, p_hat):
        self.H = self.measure_jac_function(x_hat) 
        self.M = self.measure_noise_jac_function(x_hat)
        
class UKF():
    def __init__(self, propogate_function, measurement_function, Q, R, alpha=1e-1, beta=2, kappa=0):
        self.f = propogate_function # must have input args x, u, w
        self.y = measurement_function # must have input args x, v
        self.a = alpha
        self.b = beta
        self.k = kappa
        
        if np.isscalar(Q):
            self.Q = np.array([Q])
        else:
            self.Q = Q
        
        if np.isscalar(R):
            self.R = np.array([R])
        else:
            self.R = R
        
    def predict(self, x_hat, p_hat, u=0):
        sigmas, c_weights, m_weights = self.gen_sigma_points(x_hat, p_hat)
        UT_sigmas = np.zeros((x_hat.shape[0], sigmas.shape[1]))
        
        # propogata sigma points through unscented transform
        for j in range(sigmas.shape[1]):
            sigma_x = sigmas[:x_hat.shape[0], j]
            sigma_w = sigmas[x_hat.shape[0] : x_hat.shape[0]+self.Q.shape[0], j]
            UT_sigmas[:,j] = self.f(sigma_x, u, sigma_w)
        
        # calculate a priori mean
        weighted_m_sigmas = UT_sigmas * m_weights
        x_apriori = np.sum(weighted_m_sigmas, axis=1)
        
        # calculate apriori covariance
        P_apriori = np.zeros((x_hat.shape[0], x_hat.shape[0]))
        for j in range(sigmas.shape[1]):
            dif_vector = UT_sigmas[:,j] - x_apriori
            sigma_covariance =  np.outer(dif_vector, dif_vector)
            P_apriori += sigma_covariance * c_weights[j]
        
        # log sigma points and weights in filter for measurement
        sigmas[:x_hat.shape[0], :] = UT_sigmas
        self.sigmas = sigmas
        self.m_weights = m_weights
        self.c_weights = c_weights
        
        return x_apriori, P_apriori
    
    def correct(self, y, x_apriori, P_apriori):
        # necessary so we can take the dimension of y
        if np.isscalar(y):
            y = np.array([y])
        
        measured_sigmas = np.zeros((self.R.shape[0], self.sigmas.shape[1]))
        
        # propogata sigma points through measurement function
        for j in range(self.sigmas.shape[1]):
            sigma_x = self.sigmas[:x_apriori.shape[0], j]
            sigma_v = self.sigmas[x_apriori.shape[0]+self.Q.shape[0]:, j]
            measured_sigmas[:,j] = self.y(sigma_x, sigma_v)
        
        # calculate expected measurement
        weighted_measurements = measured_sigmas * self.m_weights
        y_hat = np.sum(weighted_measurements, axis=1)
        
        # compute innovation covariance matrix
        P_y = np.zeros((y.shape[0], y.shape[0]))
        for j in range(self.sigmas.shape[1]):
            dif_vector = measured_sigmas[:,j] - y_hat
            sigma_covariance =  np.outer(dif_vector, dif_vector)
            P_y += sigma_covariance * self.c_weights[j]
        
        # compute cross-covariance matrix
        P_xy = np.zeros((x_apriori.shape[0], y.shape[0]))
        for j in range(self.sigmas.shape[1]):
            dif_y = measured_sigmas[:,j] - y_hat
            sigma_x = self.sigmas[:x_apriori.shape[0], j]
            dif_x = sigma_x - x_apriori
            sigma_covariance =  np.outer(dif_x, dif_y)
            P_xy += sigma_covariance * self.c_weights[j]
        
        if y.shape[0] == 1:
            K =  P_xy / (P_y)
            x_aposteriori = x_apriori + K @ (y - y_hat)
            P_aposteriori = P_apriori - K @ P_y @ K.T
        else:
            K = P_xy @ np.linalg.inv(P_y)
            x_aposteriori = x_apriori + K @ (y - y_hat)
            P_aposteriori = P_apriori - K @ P_y @ K.T
        
        # log kalman gain for fun
        self.K = K
        
        return x_aposteriori, P_aposteriori
        
    def gen_sigma_points(self, x_hat, p_hat):
        
        # define augmented state and covariance matrix
        dim_sigma = x_hat.shape[0] + self.Q.shape[0] + self.R.shape[0]
        sigmas = np.zeros((dim_sigma, 2*dim_sigma + 1))
        m_weights = np.zeros(sigmas.shape[1])
        c_weights = np.zeros(sigmas.shape[1])
        P_aug = scipy.linalg.block_diag(p_hat, self.Q, self.R)
        lam = self.a**2*(dim_sigma + self.k) - dim_sigma
        
        # calculate central sigma point and weights
        sigma0 = np.block([x_hat, np.zeros(sigmas.shape[0] - x_hat.shape[0])])
        sigmas[:,0] = sigma0
        m_weights[0] = lam/(dim_sigma + lam)
        c_weights[0] = lam/(dim_sigma + lam) + 1 - self.a**2 + self.b
        
        # calculate all other sigma points and weights
        for j in range(1, dim_sigma+1):
            root_P = np.linalg.cholesky((dim_sigma + lam )*P_aug) # FIXME understand why taking root of row instead of this fucked the filter 
            displacement = root_P[j-1,:]
            sigmas[:,j] = sigma0 + displacement
            sigmas[:,j+dim_sigma] = sigma0 - displacement
            m_weights[j] = 1/(2*(dim_sigma + lam))
            m_weights[j+dim_sigma] = m_weights[j]
            c_weights[j] = m_weights[j]
            c_weights[j+dim_sigma] = m_weights[j]
            
        return sigmas, c_weights, m_weights
    
class FIKS(KF):
    # this assumes u is a scalar
    def smooth(self, x_plus1_posteriori, x_posteriori, u, P_plus1_posteriori, P_posteriori):
        innovation = x_plus1_posteriori - self.F @ x_posteriori - self.G * u
        Ks = P_posteriori @ self.F.T @ np.linalg.inv(self.F @ P_posteriori @ self.F.T + self.Q)
        x_smoothed = x_posteriori + Ks @ innovation
        P_smoothed = P_posteriori + Ks @ (P_plus1_posteriori - self.F @ P_posteriori @ self.F.T - self.Q) @ Ks.T
        return x_smoothed, P_smoothed