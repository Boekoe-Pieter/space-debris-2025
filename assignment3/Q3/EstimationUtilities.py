import numpy as np
import math
from datetime import datetime, timedelta
import os
import pandas as pd
import pickle


from tudatpy.numerical_simulation import environment_setup


import TudatPropagator as prop


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


def ukf(state_parameters, measurement_data, sensor_configurations, integration_params, filter_settings, celestial_bodies):
    """Unscented Kalman Filter implementation for orbit determination."""
    
    # ----------------------------
    # FILTER INITIALIZATION
    # ----------------------------
    # Extract initial conditions
    initial_epoch = state_parameters['epoch_tdb']
    state_vector = state_parameters['state'].copy()
    covariance_matrix = state_parameters['covar'].copy()
    
    # Process noise configuration
    eci_process_noise = filter_settings['Qeci']
    ric_process_noise = filter_settings['Qric']
    param_process_noise = filter_settings.get('Qphysical', None)
    scaling_factor = filter_settings['alpha']
    max_propagation_gap = filter_settings['gap_seconds']

    state_size = len(state_vector)
    noise_dimensions = eci_process_noise.shape[0]

    # ----------------------------
    # SIGMA POINT CONFIGURATION
    # ----------------------------
    # Weighting parameters
    distribution_beta = 2.0
    distribution_kappa = 3.0 - state_size
    
    # Sigma point spread calculation
    lambda_param = scaling_factor**2 * (state_size + distribution_kappa) - state_size
    scaling_parameter = np.sqrt(state_size + lambda_param)
    
    # Weight vectors initialization
    base_weight = 1 / (2 * (state_size + lambda_param))
    covariance_weights = np.full(2 * state_size, base_weight)
    mean_weights = covariance_weights.copy()
    
    # Primary weights adjustment
    mean_weights = np.insert(mean_weights, 0, lambda_param / (state_size + lambda_param))
    covariance_weights = np.insert(covariance_weights, 0, 
                                 lambda_param/(state_size + lambda_param) + (1 - scaling_factor**2 + distribution_beta))
    weight_matrix = np.diag(covariance_weights)

    # ----------------------------
    # FILTER PROCESSING LOOP
    # ----------------------------
    results = {}
    measurement_times = measurement_data['tk_list']
    measurement_values = measurement_data['Yk_list']
    sensor_sequence = measurement_data.get('sensor_list', ['default']*len(measurement_times))
    
    for time_step in range(len(measurement_times)):
        # Time interval handling
        previous_time = initial_epoch if time_step == 0 else measurement_times[time_step-1]
        current_time = measurement_times[time_step]
        
        # State propagation
        if previous_time == current_time:
            predicted_state = state_vector.copy()
            predicted_covariance = covariance_matrix.copy()
        else:
            time_window = np.array([previous_time, current_time])
            _, predicted_state, predicted_covariance = prop.propagate_state_and_covar(
                state_vector, covariance_matrix, time_window, 
                state_parameters, integration_params, celestial_bodies, scaling_factor)
        
        # ----------------------------
        # NOISE INJECTION
        # ----------------------------
        time_difference = current_time - previous_time
        noise_transition = np.zeros((state_size, noise_dimensions))
        if time_difference <= max_propagation_gap:
            noise_transition[:noise_dimensions, :] = (time_difference**2/2) * np.eye(noise_dimensions)
            noise_transition[noise_dimensions:2*noise_dimensions, :] = time_difference * np.eye(noise_dimensions)
        
        # Reference frame transformation
        position_vector = predicted_state[:3].reshape(3,1)
        velocity_vector = predicted_state[3:6].reshape(3,1)
        combined_noise = eci_process_noise + ric2eci(position_vector, velocity_vector, ric_process_noise)
        
        # Covariance augmentation
        noise_contribution = noise_transition @ combined_noise @ noise_transition.T
        predicted_covariance += noise_contribution
        
        if param_process_noise is not None:
            param_size = param_process_noise.shape[0]
            predicted_covariance[-param_size:, -param_size:] += param_process_noise
        
        predicted_covariance = 0.5 * (predicted_covariance + predicted_covariance.T)  # Symmetrization

        # ----------------------------
        # MEASUREMENT UPDATE
        # ----------------------------
        # Sigma point regeneration
        covariance_root = np.linalg.cholesky(predicted_covariance)
        state_replicated = np.tile(predicted_state, (1, state_size))
        sigma_points = np.hstack((
            predicted_state, 
            state_replicated + scaling_parameter * covariance_root,
            state_replicated - scaling_parameter * covariance_root
        ))
        sigma_deviation = sigma_points - predicted_state @ np.ones((1, 2*state_size + 1))

        # Sensor-specific processing
        current_measurement = measurement_values[time_step]
        active_sensor = sensor_configurations[sensor_sequence[time_step]]
        
        # Measurement prediction
        predicted_measurements, measurement_noise = unscented_meas(
            current_time, sigma_points, active_sensor, celestial_bodies)
        measurement_mean = predicted_measurements @ mean_weights.T
        measurement_mean = measurement_mean.reshape(-1, 1)
        
        # Covariance calculations
        measurement_diff = predicted_measurements - measurement_mean @ np.ones((1, 2*state_size + 1))
        innovation_covariance = measurement_diff @ weight_matrix @ measurement_diff.T + measurement_noise
        cross_correlation = sigma_deviation @ weight_matrix @ measurement_diff.T
        
        # Kalman gain computation
        kalman_gain = cross_correlation @ np.linalg.inv(innovation_covariance)
        state_vector = predicted_state + kalman_gain @ (current_measurement - measurement_mean)
        
        # ----------------------------
        # COVARIANCE UPDATE
        # ----------------------------
        covariance_root_inv = np.linalg.inv(np.linalg.cholesky(predicted_covariance))
        inverse_covariance = covariance_root_inv.T @ covariance_root_inv
        covariance_term1 = np.eye(state_size) - kalman_gain @ innovation_covariance @ kalman_gain.T @ inverse_covariance
        covariance_term2 = kalman_gain @ measurement_noise @ kalman_gain.T
        covariance_matrix = covariance_term1 @ predicted_covariance @ covariance_term1.T + covariance_term2

        # ----------------------------
        # RESIDUAL CALCULATION
        # ----------------------------
        covariance_root = np.linalg.cholesky(covariance_matrix)
        state_replicated = np.tile(state_vector, (1, state_size))
        updated_sigma_points = np.hstack((
            state_vector, 
            state_replicated + scaling_parameter * covariance_root,
            state_replicated - scaling_parameter * covariance_root
        ))        
        post_update_measurements, _ = unscented_meas(current_time, updated_sigma_points, active_sensor, celestial_bodies)
        post_measurement_mean = post_update_measurements @ mean_weights.T
        residuals = current_measurement - post_measurement_mean.reshape(-1, 1)

        # Store results
        results[current_time] = {
            'state': state_vector.copy(),
            'covar': covariance_matrix.copy(),
            'resids': residuals.copy()
        }
    
    return results

###############################################################################
# Sensors and Measurements
###############################################################################


def unscented_meas(tk, chi, sensor_params, bodies):
   
    n = int(chi.shape[0])
    
    # Rotation matrices 
    earth_rotation_model = bodies.get("Earth").rotation_model
    eci2ecef = earth_rotation_model.inertial_to_body_fixed_rotation(tk)
    ecef2eci = earth_rotation_model.body_fixed_to_inertial_rotation(tk)
    
    # Sensor position calculation 
    sensor_ecef = sensor_params['sensor_ecef']
    sensor_eci = np.dot(ecef2eci, sensor_ecef)
    
    # Measurement setup 
    meas_types = sensor_params['meas_types']
    sigma_dict = sensor_params['sigma_dict']
    p = len(meas_types)
    Rk = np.zeros((p, p))
    for ii in range(p):
        mtype = meas_types[ii]
        sig = sigma_dict[mtype]
        Rk[ii,ii] = sig**2.

    # Celestial body data 
    if 'mag' in meas_types:
        sun_mag = -26.74  
        sun_x_eci = bodies.get("Sun").ephemeris.cartesian_state(tk)[:3]

    gamma_til = np.zeros((p, (2*n+1)))
    for jj in range(2*n+1):
        x, y, z = chi[0,jj], chi[1,jj], chi[2,jj]
        r_eci = np.reshape([x,y,z], (3,1))
        
        rho_eci = r_eci - sensor_eci
        rg = np.linalg.norm(rho_eci)  
        rho_hat_eci = rho_eci/rg      
        rho_hat_ecef = np.dot(eci2ecef, rho_hat_eci)
        rho_hat_enu = ecef2enu(rho_hat_ecef, sensor_ecef)
    

        if 'mag' in meas_types:
            idx_mag = meas_types.index('mag')
            Cr = chi[7,jj] if n > 7 else 1.3  
            
            # Vector calculations 
            sun_to_target = r_eci.flatten() - sun_x_eci
            observer_to_target = rho_eci.flatten()
            
            # Phase angle calculation
            cos_phase = np.dot(sun_to_target, observer_to_target)/(
                np.linalg.norm(sun_to_target)*np.linalg.norm(observer_to_target))
            cos_phase = np.clip(cos_phase, -1.0, 1.0)
            phase_angle = np.arccos(cos_phase)
            
            # Phase function formulation
            phase_func = (np.sin(phase_angle) + 
                         (np.pi-phase_angle)*np.cos(phase_angle))/np.pi**2
            
            # Magnitude calculation
            mag_value = sun_mag - 2.5*np.log10(
                (Cr - 1)*phase_func/rg**2) 
            gamma_til[idx_mag,jj] = mag_value


        if 'rg' in meas_types:
            gamma_til[meas_types.index('rg'),jj] = rg  
            
        if 'ra' in meas_types:
            ra = math.atan2(rho_hat_eci[1], rho_hat_eci[0])
            if jj == 0:
                quad = 0
                if ra > np.pi/2. and ra < np.pi: quad = 2
                if ra < -np.pi/2. and ra > -np.pi: quad = 3
            else:
                if quad == 2 and ra < 0.: ra += 2.*np.pi
                if quad == 3 and ra > 0.: ra -= 2.*np.pi
            gamma_til[meas_types.index('ra'),jj] = ra
                
        if 'dec' in meas_types:        
            gamma_til[meas_types.index('dec'),jj] = math.asin(rho_hat_eci[2])
            
        if 'az' in meas_types:
            az = math.atan2(rho_hat_enu[0], rho_hat_enu[1])
            if jj == 0:
                quad = 0
                if az > np.pi/2. and az < np.pi: quad = 2
                if az < -np.pi/2. and az > -np.pi: quad = 3
            else:
                if quad == 2 and az < 0.: az += 2.*np.pi
                if quad == 3 and az > 0.: az -= 2.*np.pi
            gamma_til[meas_types.index('az'),jj] = az
            
        if 'el' in meas_types:
            gamma_til[meas_types.index('el'),jj] = math.asin(rho_hat_enu[2])

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