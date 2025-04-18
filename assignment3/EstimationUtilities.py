import numpy as np
import math
from datetime import datetime, timedelta
import os
import pandas as pd
import pickle


from tudatpy.numerical_simulation import environment_setup


import TudatPropagator as prop
from scipy.stats import chi2


###############################################################################
# Basic I/O
###############################################################################

def read_truth_file(truth_file):
    '''
    This function reads a pickle file containing truth data for state 
    estimation.
    
    Parameters
    ------
    truth_file : string
        path and filename of pickle file containing truth data
    
    Returns
    ------
    t_truth : N element numpy array
        time in seconds since J2000
    X_truth : Nxn numpy array
        each row X_truth[k,:] corresponds to Cartesian state at time t_truth[k]
    state_params : dictionary
        propagator params
        
        fields:
            Cd: float, drag coefficient
            Cr: float, reflectivity coefficient
            area: float [m^2]
            mass: float [kg]
            sph_deg: int, spherical harmonics expansion degree for Earth
            sph_ord: int, spherical harmonics expansion order for Earth
            central_bodies: list of central bodies for propagator ["Earth"]
            bodies_to_create: list of bodies to create ["Earth", "Sun", "Moon"]
            
    '''
    
    # Load truth data
    pklFile = open(truth_file, 'rb' )
    data = pickle.load( pklFile )
    t_truth = data[0]
    X_truth = data[1]
    state_params = data[2]
    pklFile.close()
    
    return t_truth, X_truth, state_params


def read_measurement_file(meas_file):
    '''
    This function reads a pickle file containing measurement data for state 
    estimation.
    
    Parameters
    ------
    meas_file : string
        path and filename of pickle file containing measurement data
    
    Returns
    ------
    state_params : dictionary
        initial state and covariance for filter execution and propagator params
        
        fields:
            epoch_tdb: time in seconds since J2000 TDB
            state: nx1 numpy array contaiing position/velocity state in ECI [m, m/s]
            covar: nxn numpy array containing Gaussian covariance matrix [m^2, m^2/s^2]
            Cd: float, drag coefficient
            Cr: float, reflectivity coefficient
            area: float [m^2]
            mass: float [kg]
            sph_deg: int, spherical harmonics expansion degree for Earth
            sph_ord: int, spherical harmonics expansion order for Earth
            central_bodies: list of central bodies for propagator ["Earth"]
            bodies_to_create: list of bodies to create ["Earth", "Sun", "Moon"]
            
    meas_dict : dictionary
        measurement data over time for the filter 
        
        fields:
            tk_list: list of times in seconds since J2000
            Yk_list: list of px1 numpy arrays containing measurement data
            
    sensor_params : dictionary
        location, constraint, noise parameters of sensor
            
    '''

    # Load measurement data
    pklFile = open(meas_file, 'rb' )
    data = pickle.load( pklFile )
    state_params = data[0]
    sensor_params = data[1]
    meas_dict = data[2]
    pklFile.close()
    
    return state_params, meas_dict, sensor_params


###############################################################################
# Unscented Kalman Filter
###############################################################################
def ukf(state_params, meas_dict, sensor_params, int_params, filter_params, bodies,cutoff):    
    '''
    This function implements the Unscented Kalman Filter for the least
    squares cost function.

    Parameters
    ------
    state_params : dictionary
        initial state and covariance for filter execution and propagator params
        
        fields:
            epoch_tdb: epoch of state/covar [seconds since J2000]
            state: nx1 numpy array contaiing position/velocity state in ECI [m, m/s]
            covar: nxn numpy array containing Gaussian covariance matrix [m^2, m^2/s^2]
            Cd: float, drag coefficient
            Cr: float, reflectivity coefficient
            area: float [m^2]
            mass: float [kg]
            sph_deg: int, spherical harmonics expansion degree for Earth
            sph_ord: int, spherical harmonics expansion order for Earth
            central_bodies: list of central bodies for propagator ["Earth"]
            bodies_to_create: list of bodies to create ["Earth", "Sun", "Moon"]
            
    meas_dict : dictionary
        measurement data over time for the filter 
        
        fields:
            tk_list: list of times in seconds since J2000
            Yk_list: list of px1 numpy arrays containing measurement data
            
    sensor_params : dictionary
        location, constraint, noise parameters of sensor
        
    int_params : dictionary
        numerical integration parameters
        
    filter_params : dictionary
        fields:
            Qeci: 3x3 numpy array of SNC accelerations in ECI [m/s^2]
            Qric: 3x3 numpy array of SNC accelerations in RIC [m/s^2]
            alpha: float, UKF sigma point spread parameter, should be in range [1e-4, 1]
            gap_seconds: float, time in seconds between measurements for which SNC should be zeroed out, i.e., if tk-tk_prior > gap_seconds, set Q=0
            
    bodies : tudat object
        contains parameters for the environment bodies used in propagation

    Returns
    ------
    filter_output : dictionary
        output state, covariance, and post-fit residuals at measurement times
        
        indexed first by tk, then contains fields:
            state: nx1 numpy array, estimated Cartesian state vector at tk [m, m/s]
            covar: nxn numpy array, estimated covariance at tk [m^2, m^2/s^2]
            resids: px1 numpy array, measurement residuals at tk [meters and/or radians]
        
    '''
        
    # Retrieve data from input parameters
    t0 = state_params['epoch_tdb']
    Xo = state_params['state']
    Po = state_params['covar']    
    Qeci = filter_params['Qeci']
    Qric = filter_params['Qric']
    alpha = filter_params['alpha']
    gap_seconds = filter_params['gap_seconds']

    n = len(Xo)
    q = int(Qeci.shape[0])
    
    # Prior information about the distribution
    beta = 2.
    kappa = 3. - float(n)
    
    # Compute sigma point weights    
    lam = alpha**2.*(n + kappa) - n
    gam = np.sqrt(n + lam)
    Wm = 1./(2.*(n + lam)) * np.ones(2*n,)
    Wc = Wm.copy()
    Wm = np.insert(Wm, 0, lam/(n + lam))
    Wc = np.insert(Wc, 0, lam/(n + lam) + (1 - alpha**2 + beta))
    diagWc = np.diag(Wc)

    # Initialize output
    filter_output = {}

    # Measurement times
    tk_list = meas_dict['tk_list'][:cutoff]
    Yk_list = meas_dict['Yk_list'][:cutoff]

    # Number of epochs
    N = len(tk_list)
  
    # Loop over times
    Xk = Xo.copy()
    Pk = Po.copy()
    for kk in range(N):
    
        # Current and previous time
        if kk == 0:
            tk_prior = t0
        else:
            tk_prior = tk_list[kk-1]

        tk = tk_list[kk]
        
        # Propagate state and covariance
        # No prediction needed if measurement time is same as current state
        if tk_prior == tk:
            Xbar = Xk.copy()
            Pbar = Pk.copy()
        else:
            tvec = np.array([tk_prior, tk])
            dum, Xbar, Pbar = prop.propagate_state_and_covar(Xk, Pk, tvec, state_params, int_params, bodies, alpha)
       
        # State Noise Compensation
        # Zero out SNC for long time gaps
        delta_t = tk - tk_prior
        if delta_t > gap_seconds:    
            Gamma = np.zeros((n,q))
        else:
            Gamma = np.zeros((n,q))
            Gamma[0:q,:] = (delta_t**2./2) * np.eye(q)
            Gamma[q:2*q,:] = delta_t * np.eye(q)

        # Combined Q matrix (ECI and RIC components)
        # Rotate RIC to ECI and add
        rc_vect = Xbar[0:3].reshape(3,1)
        vc_vect = Xbar[3:6].reshape(3,1)
        Q = Qeci + ric2eci(rc_vect, vc_vect, Qric)
                
        # Add Process Noise to Pbar
        Pbar += np.dot(Gamma, np.dot(Q, Gamma.T))

        # Recompute sigma points to incorporate process noise
        sqP = np.linalg.cholesky(Pbar)
        Xrep = np.tile(Xbar, (1, n))
        chi_bar = np.concatenate((Xbar, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1) 
        chi_diff = chi_bar - np.dot(Xbar, np.ones((1, (2*n+1))))
        
        # Measurement Update: posterior state and covar at tk       
        # Retrieve measurement data
        Yk = Yk_list[kk]
        
        # Computed measurements and covariance
        gamma_til_k, Rk = unscented_meas(tk, chi_bar, sensor_params, bodies)
        ybar = np.dot(gamma_til_k, Wm.T)
        ybar = np.reshape(ybar, (len(ybar), 1))
        Y_diff = gamma_til_k - np.dot(ybar, np.ones((1, (2*n+1))))
        Pyy = np.dot(Y_diff, np.dot(diagWc, Y_diff.T)) + Rk
        Pxy = np.dot(chi_diff,  np.dot(diagWc, Y_diff.T))
        
        # Kalman gain and measurement update
        Kk = np.dot(Pxy, np.linalg.inv(Pyy))
        Xk = Xbar + np.dot(Kk, Yk-ybar)
        
        # Joseph form of covariance update
        cholPbar = np.linalg.inv(np.linalg.cholesky(Pbar))
        invPbar = np.dot(cholPbar.T, cholPbar)
        P1 = (np.eye(n) - np.dot(np.dot(Kk, np.dot(Pyy, Kk.T)), invPbar))
        P2 = np.dot(Kk, np.dot(Rk, Kk.T))
        P = np.dot(P1, np.dot(Pbar, P1.T)) + P2

        # Recompute measurments using final state to get resids
        sqP = np.linalg.cholesky(P)
        Xrep = np.tile(Xk, (1, n))
        chi_k = np.concatenate((Xk, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)        
        gamma_til_post, dum = unscented_meas(tk, chi_k, sensor_params, bodies)
        ybar_post = np.dot(gamma_til_post, Wm.T)
        ybar_post = np.reshape(ybar_post, (len(ybar), 1))
        
        # Post-fit residuals and updated state
        resids = Yk - ybar_post
        
        # print('')
        # print('kk', kk)
        # print('Yk', Yk)
        # print('ybar', ybar)     
        # print('resids', resids)
        
        # Store output
        filter_output[tk] = {}
        filter_output[tk]['state'] = Xk
        filter_output[tk]['covar'] = P
        filter_output[tk]['resids'] = resids
        # '''
        # residuals are slant range, Ra and Dec
        # '''
        # M_C = 0.5 * np.linalg.inv(Pyy)
        # D_C = resids.T @ M_C @ resids
        # alpha = 0.96
        # '''
        # chi2.pff gives the ritical value (threshold) for the chi-squared distribution,
        # it's used to determine whether the computed test statistic (control distance)
        # is unusually large ==> maybe a manouvre 
        # '''
        # threshold = chi2.ppf(alpha, df=3)
        # '''
        # Find the time stamp
        # '''
        # if D_C > threshold:
        #     print(f"Maneuver detected at {tk}: D_C = {float(D_C):.2f}, Threshold = {float(threshold):.2f}, alpha = {alpha} ,Sigma = {np.sqrt(D_C)[0][0]}")
        return filter_output

def ukf_manouvre(state_params, meas_dict, sensor_params, int_params, filter_params, bodies):    
    '''
    This function implements the Unscented Kalman Filter for the least
    squares cost function.

    Parameters
    ------
    state_params : dictionary
        initial state and covariance for filter execution and propagator params
        
        fields:
            epoch_tdb: epoch of state/covar [seconds since J2000]
            state: nx1 numpy array contaiing position/velocity state in ECI [m, m/s]
            covar: nxn numpy array containing Gaussian covariance matrix [m^2, m^2/s^2]
            Cd: float, drag coefficient
            Cr: float, reflectivity coefficient
            area: float [m^2]
            mass: float [kg]
            sph_deg: int, spherical harmonics expansion degree for Earth
            sph_ord: int, spherical harmonics expansion order for Earth
            central_bodies: list of central bodies for propagator ["Earth"]
            bodies_to_create: list of bodies to create ["Earth", "Sun", "Moon"]
            
    meas_dict : dictionary
        measurement data over time for the filter 
        
        fields:
            tk_list: list of times in seconds since J2000
            Yk_list: list of px1 numpy arrays containing measurement data
            
    sensor_params : dictionary
        location, constraint, noise parameters of sensor
        
    int_params : dictionary
        numerical integration parameters
        
    filter_params : dictionary
        fields:
            Qeci: 3x3 numpy array of SNC accelerations in ECI [m/s^2]
            Qric: 3x3 numpy array of SNC accelerations in RIC [m/s^2]
            alpha: float, UKF sigma point spread parameter, should be in range [1e-4, 1]
            gap_seconds: float, time in seconds between measurements for which SNC should be zeroed out, i.e., if tk-tk_prior > gap_seconds, set Q=0
            
    bodies : tudat object
        contains parameters for the environment bodies used in propagation

    Returns
    ------
    filter_output : dictionary
        output state, covariance, and post-fit residuals at measurement times
        
        indexed first by tk, then contains fields:
            state: nx1 numpy array, estimated Cartesian state vector at tk [m, m/s]
            covar: nxn numpy array, estimated covariance at tk [m^2, m^2/s^2]
            resids: px1 numpy array, measurement residuals at tk [meters and/or radians]
        
    '''
        
    # Retrieve data from input parameters
    t0 = state_params['epoch_tdb']
    Xo = state_params['state']
    Po = state_params['covar']    
    Qeci = filter_params['Qeci']
    Qric = filter_params['Qric']
    alpha = filter_params['alpha']
    gap_seconds = filter_params['gap_seconds']

    n = len(Xo)
    q = int(Qeci.shape[0])
    
    # Prior information about the distribution
    beta = 2.
    kappa = 3. - float(n)
    
    # Compute sigma point weights    
    lam = alpha**2.*(n + kappa) - n
    gam = np.sqrt(n + lam)
    Wm = 1./(2.*(n + lam)) * np.ones(2*n,)
    Wc = Wm.copy()
    Wm = np.insert(Wm, 0, lam/(n + lam))
    Wc = np.insert(Wc, 0, lam/(n + lam) + (1 - alpha**2 + beta))
    diagWc = np.diag(Wc)

    # Initialize output
    filter_output = {}

    # Measurement times
    tk_list = meas_dict['tk_list']
    Yk_list = meas_dict['Yk_list']

    # Number of epochs
    N = len(tk_list)
  
    # Loop over times
    Xk = Xo.copy()
    Pk = Po.copy()
    for kk in range(N):
    
        # Current and previous time
        if kk == 0:
            tk_prior = t0
        else:
            tk_prior = tk_list[kk-1]

        tk = tk_list[kk]
        
        # Propagate state and covariance
        # No prediction needed if measurement time is same as current state
        if tk_prior == tk:
            Xbar = Xk.copy()
            Pbar = Pk.copy()
        else:
            tvec = np.array([tk_prior, tk])
            dum, Xbar, Pbar = prop.propagate_state_and_covar(Xk, Pk, tvec, state_params, int_params, bodies, alpha)
       
        # State Noise Compensation
        # Zero out SNC for long time gaps
        delta_t = tk - tk_prior
        if delta_t > gap_seconds:    
            Gamma = np.zeros((n,q))
        else:
            Gamma = np.zeros((n,q))
            Gamma[0:q,:] = (delta_t**2./2) * np.eye(q)
            Gamma[q:2*q,:] = delta_t * np.eye(q)

        # Combined Q matrix (ECI and RIC components)
        # Rotate RIC to ECI and add
        rc_vect = Xbar[0:3].reshape(3,1)
        vc_vect = Xbar[3:6].reshape(3,1)
        Q = Qeci + ric2eci(rc_vect, vc_vect, Qric)
                
        # Add Process Noise to Pbar
        Pbar += np.dot(Gamma, np.dot(Q, Gamma.T))

        # Recompute sigma points to incorporate process noise
        sqP = np.linalg.cholesky(Pbar)
        Xrep = np.tile(Xbar, (1, n))
        chi_bar = np.concatenate((Xbar, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1) 
        chi_diff = chi_bar - np.dot(Xbar, np.ones((1, (2*n+1))))
        
        # Measurement Update: posterior state and covar at tk       
        # Retrieve measurement data
        Yk = Yk_list[kk]
        
        # Computed measurements and covariance
        gamma_til_k, Rk = unscented_meas(tk, chi_bar, sensor_params, bodies)
        ybar = np.dot(gamma_til_k, Wm.T)
        ybar = np.reshape(ybar, (len(ybar), 1))
        Y_diff = gamma_til_k - np.dot(ybar, np.ones((1, (2*n+1))))
        Pyy = np.dot(Y_diff, np.dot(diagWc, Y_diff.T)) + Rk
        Pxy = np.dot(chi_diff,  np.dot(diagWc, Y_diff.T))
        
        # Kalman gain and measurement update
        Kk = np.dot(Pxy, np.linalg.inv(Pyy))
        Xk = Xbar + np.dot(Kk, Yk-ybar)
        
        # Joseph form of covariance update
        cholPbar = np.linalg.inv(np.linalg.cholesky(Pbar))
        invPbar = np.dot(cholPbar.T, cholPbar)
        P1 = (np.eye(n) - np.dot(np.dot(Kk, np.dot(Pyy, Kk.T)), invPbar))
        P2 = np.dot(Kk, np.dot(Rk, Kk.T))
        P = np.dot(P1, np.dot(Pbar, P1.T)) + P2

        # Recompute measurments using final state to get resids
        sqP = np.linalg.cholesky(P)
        Xrep = np.tile(Xk, (1, n))
        chi_k = np.concatenate((Xk, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)        
        gamma_til_post, dum = unscented_meas(tk, chi_k, sensor_params, bodies)
        ybar_post = np.dot(gamma_til_post, Wm.T)
        ybar_post = np.reshape(ybar_post, (len(ybar), 1))
        
        # Post-fit residuals and updated state
        resids = Yk - ybar_post
        
        # print('')
        # print('kk', kk)
        # print('Yk', Yk)
        # print('ybar', ybar)     
        # print('resids', resids)
        
        # Store output
        filter_output[tk] = {}
        filter_output[tk]['state'] = Xk
        filter_output[tk]['covar'] = P
        filter_output[tk]['resids'] = resids


        '''
        residuals are slant range, Ra and Dec
        '''
        M_C = 0.5 * np.linalg.inv(Pyy)
        D_C = resids.T @ M_C @ resids
        alpha = 0.99
        '''
        chi2.pff gives the ritical value (threshold) for the chi-squared distribution,
        it's used to determine whether the computed test statistic (control distance)
        is unusually large ==> maybe a manouvre 
        '''
        threshold = chi2.ppf(alpha, df=3)
        '''
        Find the time stamp
        '''
        if D_C > threshold:
            print(f"Maneuver detected at {tk}: D_C = {float(D_C):.2f}, Threshold = {float(threshold):.2f}, alpha = {alpha} ,Sigma = {np.sqrt(D_C)[0][0]}")
            return filter_output, tk


###############################################################################
# Sensors and Measurements
###############################################################################


def unscented_meas(tk, chi, sensor_params, bodies):
    '''
    This function computes the measurement sigma point matrix.
    
    Parameters
    ------
    tk : float
        time in seconds since J2000
    chi : nx(2n+1) numpy array
        state sigma point matrix
    sensor_params : dictionary
        location, constraint, noise parameters of sensor
        
    Returns
    ------
    gamma_til : px(2n+1) numpy array
        measurement sigma point matrix
    Rk : pxp numpy array
        measurement noise covariance
        
    '''
    
    # Number of states
    n = int(chi.shape[0])
    
    # Rotation matrices
    earth_rotation_model = bodies.get("Earth").rotation_model
    eci2ecef = earth_rotation_model.inertial_to_body_fixed_rotation(tk)
    ecef2eci = earth_rotation_model.body_fixed_to_inertial_rotation(tk)
    
    # Compute sensor position in ECI
    sensor_ecef = sensor_params['sensor_ecef']
    sensor_eci = np.dot(ecef2eci, sensor_ecef)
    
    # Measurement information    
    meas_types = sensor_params['meas_types']
    sigma_dict = sensor_params['sigma_dict']
    p = len(meas_types)
    Rk = np.zeros((p, p))
    for ii in range(p):
        mtype = meas_types[ii]
        sig = sigma_dict[mtype]
        Rk[ii,ii] = sig**2.
    
    # Compute transformed sigma points
    gamma_til = np.zeros((p, (2*n+1)))
    for jj in range(2*n+1):
        
        x = chi[0,jj]
        y = chi[1,jj]
        z = chi[2,jj]
        
        # Object location in ECI
        r_eci = np.reshape([x,y,z], (3,1))
        
        # Compute range and line of sight vector
        rho_eci = r_eci - sensor_eci
        rg = np.linalg.norm(rho_eci)
        rho_hat_eci = rho_eci/rg
        
        # Rotate to ENU frame
        rho_hat_ecef = np.dot(eci2ecef, rho_hat_eci)
        rho_hat_enu = ecef2enu(rho_hat_ecef, sensor_ecef)
        
        if 'rg' in meas_types:
            rg_ind = meas_types.index('rg')
            gamma_til[rg_ind,jj] = rg
            
        if 'ra' in meas_types:
        
            ra = math.atan2(rho_hat_eci[1], rho_hat_eci[0]) # rad        
        
            # Store quadrant info of mean sigma point        
            if jj == 0:
                quad = 0
                if ra > np.pi/2. and ra < np.pi:
                    quad = 2
                if ra < -np.pi/2. and ra > -np.pi:
                    quad = 3
                    
            # Check and update quadrant of subsequent sigma points
            else:
                if quad == 2 and ra < 0.:
                    ra += 2.*np.pi
                if quad == 3 and ra > 0.:
                    ra -= 2.*np.pi
                    
            ra_ind = meas_types.index('ra')
            gamma_til[ra_ind,jj] = ra
                
        if 'dec' in meas_types:        
            dec = math.asin(rho_hat_eci[2])  # rad
            dec_ind = meas_types.index('dec')
            gamma_til[dec_ind,jj] = dec
            
        if 'az' in meas_types:
            az = math.atan2(rho_hat_enu[0], rho_hat_enu[1])  # rad 
            
            # Store quadrant info of mean sigma point        
            if jj == 0:
                quad = 0
                if az > np.pi/2. and az < np.pi:
                    quad = 2
                if az < -np.pi/2. and az > -np.pi:
                    quad = 3
                    
            # Check and update quadrant of subsequent sigma points
            else:
                if quad == 2 and az < 0.:
                    az += 2.*np.pi
                if quad == 3 and az > 0.:
                    az -= 2.*np.pi
                    
            az_ind = meas_types.index('az')
            gamma_til[az_ind,jj] = az
            
        if 'el' in meas_types:
            el = math.asin(rho_hat_enu[2])  # rad
            el_ind = meas_types.index('el')
            gamma_til[el_ind,jj] = el


    return gamma_til, Rk


def compute_measurement(tk, X, sensor_params, bodies=None):
    '''
    This function be used to compute a measurement given an input state vector
    and time.
    
    Parameters
    ------
    tk : float
        time in seconds since J2000
    X : nx1 numpy array
        Cartesian state vector [m, m/s]
    sensor_params : dictionary
        location, constraint, noise parameters of sensor
        
    Returns
    ------
    Y : px1 numpy array
        computed measurements for given state and sensor
    
    '''
    
    if bodies is None:
        body_settings = environment_setup.get_default_body_settings(
            ["Earth"],
            "Earth",
            "J2000")
        bodies = environment_setup.create_system_of_bodies(body_settings)
        
    # Rotation matrices
    earth_rotation_model = bodies.get("Earth").rotation_model
    eci2ecef = earth_rotation_model.inertial_to_body_fixed_rotation(tk)
    ecef2eci = earth_rotation_model.body_fixed_to_inertial_rotation(tk)
        
    # Retrieve measurement types
    meas_types = sensor_params['meas_types']
    
    # Compute station location in ECI    
    sensor_ecef = sensor_params['sensor_ecef']
    sensor_eci = np.dot(ecef2eci, sensor_ecef)    
    
    # Object location in ECI
    r_eci = X[0:3].reshape(3,1)
    
    # Compute range and line of sight vector
    rg = np.linalg.norm(r_eci - sensor_eci)
    rho_hat_eci = (r_eci - sensor_eci)/rg
    
    # Rotate to ENU frame
    rho_hat_ecef = np.dot(eci2ecef, rho_hat_eci)
    rho_hat_enu = ecef2enu(rho_hat_ecef, sensor_ecef)
    
    # Loop over measurement types
    Y = np.zeros((len(meas_types),1))
    ii = 0
    for mtype in meas_types:
        
        if mtype == 'rg':
            Y[ii] = rg  # m
            
        elif mtype == 'ra':
            Y[ii] = math.atan2(rho_hat_eci[1], rho_hat_eci[0]) # rad
            
        elif mtype == 'dec':
            Y[ii] = math.asin(rho_hat_eci[2])  # rad
    
        elif mtype == 'az':
            Y[ii] = math.atan2(rho_hat_enu[0], rho_hat_enu[1])  # rad  
            # if Y[ii] < 0.:
            #     Y[ii] += 2.*np.pi
            
        elif mtype == 'el':
            Y[ii] = math.asin(rho_hat_enu[2])  # rad
            
            
        ii += 1
            
            
    return Y


###############################################################################
# Coordinate Frames
###############################################################################

def ecef2enu(r_ecef, r_site):
    '''
    This function converts the coordinates of a position vector from
    the ECEF to ENU frame.

    Parameters
    ------
    r_ecef : 3x1 numpy array
      position vector in ECEF [m]
    r_site : 3x1 numpy array
      station position vector in ECEF [m]

    Returns
    ------
    r_enu : 3x1 numpy array
      position vector in ENU [m]
    '''

    # Compute lat,lon,ht of ground station
    lat, lon, ht = ecef2latlonht(r_site)

    # Compute rotation matrix
    lat1 = math.pi/2 - lat
    lon1 = math.pi/2 + lon

    R1 = np.array([[1.,               0.,             0.],
                   [0.,   math.cos(lat1), math.sin(lat1)],
                   [0.,  -math.sin(lat1), math.cos(lat1)]])

    R3 = np.array([[math.cos(lon1),  math.sin(lon1), 0.],
                   [-math.sin(lon1), math.cos(lon1), 0.],
                   [0.,              0.,             1.]])

    R = np.dot(R1, R3)

    r_enu = np.dot(R, r_ecef)

    return r_enu


def enu2ecef(r_enu, r_site):
    '''
    This function converts the coordinates of a position vector from
    the ENU to ECEF frame.

    Parameters
    ------
    r_enu : 3x1 numpy array
      position vector in ENU [m]
    r_site : 3x1 numpy array
      station position vector in ECEF [m]

    Returns
    ------
    r_ecef : 3x1 numpy array
      position vector in ECEF
    '''

    # Compute lat,lon,ht of ground station
    lat, lon, ht = ecef2latlonht(r_site)

    # Compute rotation matrix
    lat1 = math.pi/2 - lat
    lon1 = math.pi/2 + lon

    R1 = np.array([[1.,               0.,             0.],
                   [0.,   math.cos(lat1), math.sin(lat1)],
                   [0.,  -math.sin(lat1), math.cos(lat1)]])

    R3 = np.array([[math.cos(lon1),   math.sin(lon1), 0.],
                   [-math.sin(lon1),  math.cos(lon1), 0.],
                   [0.,                           0., 1.]])

    R = np.dot(R1, R3)

    R2 = R.T

    r_ecef = np.dot(R2, r_enu)

    return r_ecef


def ecef2latlonht(r_ecef):
    '''
    This function converts the coordinates of a position vector from
    the ECEF frame to geodetic latitude, longitude, and height.

    Parameters
    ------
    r_ecef : 3x1 numpy array
      position vector in ECEF [m]

    Returns
    ------
    lat : float
      latitude [rad] [-pi/2,pi/2]
    lon : float
      longitude [rad] [-pi,pi]
    ht : float
      height [m]
    '''

    # WGS84 Data (Pratap and Misra P. 103)
    a = 6378137.0   # m
    rec_f = 298.257223563

    # Get components from position vector
    x = float(r_ecef[0])
    y = float(r_ecef[1])
    z = float(r_ecef[2])

    # Compute longitude
    f = 1./rec_f
    e = np.sqrt(2.*f - f**2.)
    lon = math.atan2(y, x)

    # Iterate to find height and latitude
    p = np.sqrt(x**2. + y**2.)  # m
    lat = 0.
    lat_diff = 1.
    tol = 1e-12

    while abs(lat_diff) > tol:
        lat0 = float(lat)  # rad
        N = a/np.sqrt(1 - e**2*(math.sin(lat0)**2))  # km
        ht = p/math.cos(lat0) - N
        lat = math.atan((z/p)/(1 - e**2*(N/(N + ht))))
        lat_diff = lat - lat0


    return lat, lon, ht


def latlonht2ecef(lat, lon, ht):
    '''
    This function converts geodetic latitude, longitude and height
    to a position vector in ECEF.

    Parameters
    ------
    lat : float
      geodetic latitude [rad]
    lon : float
      geodetic longitude [rad]
    ht : float
      geodetic height [m]

    Returns
    ------
    r_ecef = 3x1 numpy array
      position vector in ECEF [m]
    '''
    
    # WGS84 Data (Pratap and Misra P. 103)
    Re = 6378137.0   # m
    rec_f = 298.257223563

    # Compute flattening and eccentricity
    f = 1/rec_f
    e = np.sqrt(2*f - f**2)

    # Compute ecliptic plane and out of plane components
    C = Re/np.sqrt(1 - e**2*math.sin(lat)**2)
    S = Re*(1 - e**2)/np.sqrt(1 - e**2*math.sin(lat)**2)

    rd = (C + ht)*math.cos(lat)
    rk = (S + ht)*math.sin(lat)

    # Compute ECEF position vector
    r_ecef = np.array([[rd*math.cos(lon)], [rd*math.sin(lon)], [rk]])

    return r_ecef


def eci2ric(rc_vect, vc_vect, Q_eci=[]):
    '''
    This function computes the rotation from ECI to RIC and rotates input
    Q_eci (vector or matrix) to RIC.

    Parameters
    ------
    rc_vect : 3x1 numpy array
      position vector of chief (or truth) orbit in ECI
    vc_vect : 3x1 numpy array
      velocity vector of chief (or truth) orbit in ECI
    Q_eci : 3x1 or 3x3 numpy array
      vector or matrix in ECI

    Returns
    ------
    Q_ric : 3x1 or 3x3 numpy array
      vector or matrix in RIC
    '''
    
    # Reshape inputs
    rc_vect = rc_vect.reshape(3,1)
    vc_vect = vc_vect.reshape(3,1)

    # Compute transformation matrix to Hill (RIC) frame
    rc = np.linalg.norm(rc_vect)
    OR = rc_vect/rc
    h_vect = np.cross(rc_vect, vc_vect, axis=0)
    h = np.linalg.norm(h_vect)
    OH = h_vect/h
    OT = np.cross(OH, OR, axis=0)

    ON = np.concatenate((OR.T, OT.T, OH.T))

    # Rotate Q_eci as appropriate for vector or matrix
    if len(Q_eci) == 0:
        Q_ric = ON
    elif np.size(Q_eci) == 3:
        Q_eci = Q_eci.reshape(3,1)
        Q_ric = np.dot(ON, Q_eci)
    else:
        Q_ric = np.dot(np.dot(ON, Q_eci), ON.T)

    return Q_ric


def ric2eci(rc_vect, vc_vect, Q_ric=[]):
    '''
    This function computes the rotation from RIC to ECI and rotates input
    Q_ric (vector or matrix) to ECI.

    Parameters
    ------
    rc_vect : 3x1 numpy array
      position vector of chief (or truth) orbit in ECI
    vc_vect : 3x1 numpy array
      velocity vector of chief (or truth) orbit in ECI
    Q_ric : 3x1 or 3x3 numpy array
      vector or matrix in RIC

    Returns
    ------
    Q_ric : 3x1 or 3x3 numpy array
      vector or matrix in ECI
    '''
    
    # Reshape inputs
    rc_vect = rc_vect.reshape(3,1)
    vc_vect = vc_vect.reshape(3,1)

    # Compute transformation matrix to Hill (RIC) frame
    rc = np.linalg.norm(rc_vect)
    OR = rc_vect/rc
    h_vect = np.cross(rc_vect, vc_vect, axis=0)
    h = np.linalg.norm(h_vect)
    OH = h_vect/h
    OT = np.cross(OH, OR, axis=0)

    ON = np.concatenate((OR.T, OT.T, OH.T))
    NO = ON.T

    # Rotate Qin as appropriate for vector or matrix
    if len(Q_ric) == 0:
        Q_eci = NO
    elif np.size(Q_ric) == 3:
        Q_eci = np.dot(NO, Q_ric)
    else:
        Q_eci = np.dot(np.dot(NO, Q_ric), NO.T)

    return Q_eci