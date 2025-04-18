import numpy as np
import os
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import pickle

import EstimationUtilities as EstUtil
import TudatPropagator as prop
import ConjunctionUtilities as ConjUtil

from mpl_toolkits.mplot3d import Axes3D 
from tudatpy import constants

from matplotlib.animation import FuncAnimation
from scipy.optimize import least_squares

from scipy.interpolate import interp1d
np.set_printoptions(linewidth=200)

'''
Load data and we receive:
    epoch_tdb : seconds since J2000 in TDB 
    state : 6x1 numpy array, Cartesian position and velocity in ECI 
    covar : 6x6 numpy array, covariance matrix associated with state
    mass : float [kg]
    area : float [m^2]
    Cd : float, drag coefficient
    Cr : float, SRP coefficient
'''
print(f'''--------------------------------------------------------------------------------------------------------------''')
print('Loading Data...')
estimated_rso_catalog = 'assignment3/data/group1/estimated_rso_catalog.pkl'
means_maneuver = 'assignment3/data/group1/q2_meas_maneuver_91000.pkl'
rso_dict = ConjUtil.read_catalog_file(estimated_rso_catalog)
state_params, meas_dict, sensor_params = EstUtil.read_measurement_file(means_maneuver)
'''
Plots for the radar measurments to find out what we are deaing with
'''
Yk_array = np.vstack([y.flatten() for y in meas_dict['Yk_list']])
slant_range = Yk_array[:, 0]
Ra = Yk_array[:, 1]
Dec = Yk_array[:, 2]
time_days_2 = (meas_dict['tk_list'] - meas_dict['tk_list'][0]) / constants.JULIAN_DAY
time_minutes_2 = time_days_2*24*3600

def plot_measurements(slant_range,Ra,Dec,time_minutes_2):
    print(f'''--------------------------------------------------------------------------------------------------------------''')
    print('Plottig radar measurements....')
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axs[0].scatter(time_minutes_2, slant_range, color='tab:blue', s=15, label='Slant Range')
    axs[0].set_ylabel("Slant Range [m]", fontsize=12)
    axs[0].legend()
    axs[0].grid(True)

    axs[1].scatter(time_minutes_2, np.rad2deg(Ra), color='tab:orange', s=15, label='RA [deg]')
    axs[1].scatter(time_minutes_2, np.rad2deg(Dec), color='tab:green', s=15, label='Dec [deg]')
    axs[1].set_ylabel("Angle [deg]", fontsize=12)
    axs[1].set_xlabel("Time [minutes]", fontsize=12)
    axs[1].legend()
    axs[1].grid(True)

    plt.suptitle("Measurements Over Time", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('assignment3/images/measurements.png',dpi=300)
    plt.show()

# plot_measurements(slant_range,Ra,Dec,time_minutes_2)

'''
Define propagation data
'''
propagated_data = {}
t0 = (datetime(2025, 4, 1, 12, 0, 0) - datetime(2000, 1, 1, 12, 0, 0)).total_seconds()
t_vec = np.array([t0,t0+24*3600])
Epoch_start_date = constants.JULIAN_DAY_ON_J2000+t0/constants.JULIAN_DAY

int_params = {}
int_params['tudat_integrator'] = 'rkf78'
int_params['step'] = 10.
int_params['max_step'] = 1000.
int_params['min_step'] = 1e-3
int_params['rtol'] = 1e-12
int_params['atol'] = 1e-12

current_obj = rso_dict[91000]
X0 = current_obj['state']
rso_params = {
    'mass': current_obj['mass'],
    'area': current_obj['area'],
    'Cd': current_obj['Cd'],
    'Cr': current_obj['Cr'],
    'sph_deg': 8,
    'sph_ord': 8,
    'central_bodies': ['Earth'],
    'bodies_to_create': ['Earth', 'Sun', 'Moon']
}
'''
Estimate orbital parameters for the gaps between data with the UKF filter, 
find the manouvre via a costs function and stop the filter when found.
'''
# Setup filter parameters such as process noise
Qeci = 1e-12*np.diag([1., 1., 1.])
Qric = 1e-12*np.diag([1., 1., 1.])

filter_params = {}
filter_params['Qeci'] = Qeci
filter_params['Qric'] = Qric
filter_params['alpha'] = 1.
filter_params['gap_seconds'] = 600.

int_params = {}
int_params['tudat_integrator'] = 'rkf78'
int_params['step'] = 10.
int_params['max_step'] = 1000.
int_params['min_step'] = 1e-3
int_params['rtol'] = 1e-12
int_params['atol'] = 1e-12

bodies_to_create = ['Earth', 'Sun', 'Moon']
bodies = prop.tudat_initialize_bodies(bodies_to_create)
'''
Perform the kalman filter for data visualization
'''
# print(f'''--------------------------------------------------------------------------------------------------------------''')
# print('Running UKF for initial data visualization....')
# filter_output= EstUtil.ukf(state_params, meas_dict, sensor_params, int_params, filter_params, bodies,cutoff=410)
# time_steps = list(filter_output.keys())

# residuals = []
# covariance = []
# state_list = []

# for tk in time_steps:
#     resids = filter_output[tk]['resids']
#     covar = filter_output[tk]['covar'] 
#     states = filter_output[tk]['state'] 
#     residuals.append(resids.flatten()) 
#     covariance.append(covar) 
#     state_list.append(states.flatten()) 

def plot_kalman(meas_dict,states_array,residuals,covariance,cuttoff):
    '''
    plots for the kaleman filter
    '''
    print(f'''--------------------------------------------------------------------------------------------------------------''')
    print('Plottig UKF measurements....')
    time_days = (meas_dict['tk_list'][:cuttoff] - meas_dict['tk_list'][0]) / constants.JULIAN_DAY
    time_minutes = time_days*24*3600
    pos_norm = np.linalg.norm(states_array[:, :3], axis=1)
    vel_norm = np.linalg.norm(states_array[:, 3:], axis=1)

    plt.figure(figsize=(10, 5))
    plt.scatter(time_minutes,pos_norm, label='Position Norm') 
    plt.title("Position Norm Over iteration")
    plt.xlabel("time [Sec]")
    plt.ylabel("||r|| (residual norm)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('assignment3/images/position_norm.png',dpi=300)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.scatter(time_minutes,vel_norm, label='Velocity Norm') 
    plt.title("Velocity Norm Over iteration")
    plt.xlabel("time [Sec]")
    plt.ylabel("||V||")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('assignment3/images/velocity_norm.png',dpi=300)
    plt.show()

    residuals = np.array(residuals)
    labels = ["Slant Range Residual", "RA Residual (deg)", "Dec Residual (deg)"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axs[0].scatter(time_minutes, residuals[:, 0], label=labels[0], color=colors[0], s=15, alpha=0.7)
    axs[0].set_ylabel("Slant Range [m]", fontsize=12)
    axs[0].legend()
    axs[0].grid(True)

    axs[1].scatter(time_minutes, np.rad2deg(residuals[:, 1]), label=labels[1], color=colors[1], s=15, alpha=0.7)
    axs[1].scatter(time_minutes, np.rad2deg(residuals[:, 2]), label=labels[2], color=colors[2], s=15, alpha=0.7)
    axs[1].set_ylabel("Angle [deg]", fontsize=12)
    axs[1].set_xlabel("Time [minutes]", fontsize=12)
    axs[1].legend()
    axs[1].grid(True)
    plt.legend()
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('assignment3/images/residual.png',dpi=300)
    plt.show()

    labels = ["Slant Range Residual", "RA Residual (deg)", "Dec Residual (deg)"]
    colors = ["tab:blue", "tab:orange", "tab:green"]
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axs[0].scatter(time_minutes, residuals[:, 0], label=labels[0], color=colors[0], s=15, alpha=0.7)
    axs[0].set_ylabel("Slant Range [m]", fontsize=12)
    axs[0].legend()
    axs[0].grid(True)

    axs[1].scatter(time_minutes, np.rad2deg(residuals[:, 1]), label=labels[1], color=colors[1], s=15, alpha=0.7)
    axs[1].scatter(time_minutes, np.rad2deg(residuals[:, 2]), label=labels[2], color=colors[2], s=15, alpha=0.7)
    axs[1].set_ylabel("Angle [deg]", fontsize=12)
    axs[1].set_xlabel("Time [minutes]", fontsize=12)
    axs[1].legend()
    axs[1].grid(True)
    axs[1].set_yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('assignment3/images/residual_log.png',dpi=300)
    plt.show()

    innovation_norm = np.linalg.norm(residuals, axis=1)
    plt.figure(figsize=(10, 5))
    plt.scatter(time_minutes,innovation_norm, label='Innovation Norm') 
    plt.title("Innovation Norm Over Time")
    plt.xlabel("time [Sec]")
    plt.ylabel("||Yk - Ȳk|| (residual norm)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('assignment3/images/innovation_norm.png',dpi=300)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.scatter(time_minutes,innovation_norm, label='Innovation Norm') 
    plt.title("Innovation Norm Over Time")
    plt.xlabel("time [Sec]")
    plt.ylabel("||Yk - Ȳk|| (residual norm)")
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('assignment3/images/innovation_norm_log.png',dpi=300)
    plt.show()
    '''
    The covariance trace is the sum of the variances annd shows if the uncertainty is growing
    '''

    cov_trace = [np.trace(P_k) for P_k in covariance]  
    plt.figure(figsize=(10, 5))
    plt.scatter(time_minutes,cov_trace)
    plt.title("Covariance Matrix Trace Over Time")
    plt.xlabel("time [Sec]")
    plt.ylabel("Trace(P)")
    plt.grid(True)
    plt.savefig('assignment3/images/covariance trace.png',dpi=300)
    plt.show()

    stds = np.array([np.sqrt(np.diag(P)) for P in covariance])  
    plt.figure(figsize=(10, 5))
    for i in range(stds.shape[1]):
        plt.scatter(time_minutes,stds[:, i], label=f"σ_{i}")

    plt.title("Standard Deviation of States Over Time")
    plt.xlabel("time [Sec]")
    plt.ylabel("Standard Deviation")
    plt.legend()
    plt.grid(True)
    plt.savefig('assignment3/images/deviation.png',dpi=300)
    plt.show()

# plot_kalman(meas_dict,np.array(state_list),residuals,covariance,cuttoff=410)

'''
Perform the kalman filter for maneouvre detection
'''
print(f'''--------------------------------------------------------------------------------------------------------------''')
print('Analysing for possible Manoeuvre....')
filter_output,manoeuvre_epoch = EstUtil.ukf_manouvre(state_params, meas_dict, sensor_params, int_params, filter_params, bodies)
print(f'''--------------------------------------------------------------------------------------------------------------''')
print(f'Manoeuvre happend at EPOCH: {manoeuvre_epoch}')
print(f'Corresponding Calender date: {timedelta(seconds=(manoeuvre_epoch))+datetime(2000, 1, 1, 12, 0, 0)}')
print(f'Which is {((timedelta(seconds=(manoeuvre_epoch))+datetime(2000, 1, 1, 12, 0, 0)-datetime(2025, 4, 1, 12, 0, 0)).total_seconds())/3600} hours after {datetime(2025, 4, 1, 12, 0, 0)}')
print(f'''--------------------------------------------------------------------------------------------------------------''')
time_steps = list(filter_output.keys())

residuals = []
covariance = []
state_list = []
manoeuvre = 0

for tk in time_steps:
    resids = filter_output[tk]['resids']
    covar = filter_output[tk]['covar'] 
    states = filter_output[tk]['state'] 
    residuals.append(resids.flatten()) 
    covariance.append(covar) 
    state_list.append(states.flatten()) 

# plot_kalman(meas_dict,np.array(state_list),residuals,covariance,cuttoff=-1)
manouvre_begin_state  = filter_output[manoeuvre_epoch]['state'] 

'''
Least square method to construct the post mannouvre orbit
 - get the initial position and intermediate positions from the radar
 - get the initital velocity via lambert targerter
 - LSQ method to find positions and velocities to 'patch' the orbit
'''

print('Converting Ra,Dec,Slant range to ECI satellite positions....')
print(f'''--------------------------------------------------------------------------------------------------------------''')
def compute_r_eci_from_radar(Yk, sensor_params, tk, bodies):
    sensor_ecef = sensor_params['sensor_ecef']
    ecef2eci = bodies.get("Earth").rotation_model.body_fixed_to_inertial_rotation(tk)
    sensor_eci = np.dot(ecef2eci, sensor_ecef).flatten()
    rg, ra, dec = Yk.flatten()
    rho_hat_eci = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    return sensor_eci + rg * rho_hat_eci

for idx, tk in enumerate(meas_dict['tk_list']):
    if tk >= manoeuvre_epoch:
        break

#removing all radar data before the maneouvre
meas_dict['tk_list'] = meas_dict['tk_list'][idx:]
meas_dict['Yk_list'] = meas_dict['Yk_list'][idx:]


dt_list = np.diff(meas_dict['tk_list'])
jump_indices = np.where(dt_list > 10)[0]
first_jump = jump_indices[0]

meas_dict['tk_list'] = meas_dict['tk_list'][first_jump+1:]
meas_dict['Yk_list'] = meas_dict['Yk_list'][first_jump+1:]
r_eci_list = []

#removing all radar data until the first jump (thus from maneouvre data to the first new data of the new orbit)
for i, t in enumerate(meas_dict['tk_list']):
    Yk = meas_dict['Yk_list'][i]
    r_eci = compute_r_eci_from_radar(Yk, sensor_params, t, bodies)
    r_eci_list.append(r_eci)
r_eci_array = np.array(r_eci_list)
tk_array = np.array(meas_dict['tk_list'])

dt_list = np.diff(meas_dict['tk_list'])
jump_indices = np.where(dt_list > 10)[0]
print('Computing lambert targetter for satellite states....')
print(f'''--------------------------------------------------------------------------------------------------------------''')
lambert_first_arc_initial_state,lambert_first_arc_final_state = prop.lambert_targerter(
        r_eci_array[0],
        r_eci_array[jump_indices[0]], 
        tk_array[jump_indices[0]],
        tk_array[0],
        bodies
    )
lambert_second_arc_initial_state,lambert_second_arc_final_state = prop.lambert_targerter(
        r_eci_array[jump_indices[0]+1],
        r_eci_array[jump_indices[1]],
        tk_array[jump_indices[1]],
        tk_array[jump_indices[0]+1],
        bodies
    )
lambert_third_arc_initial_state,lambert_third_arc_final_state= prop.lambert_targerter(
        r_eci_array[jump_indices[1]+1],
        r_eci_array[-1],
        tk_array[-1],
        tk_array[jump_indices[1]+1],
        bodies
    )
print('First lambert targetter')
print(f'First lambert state 0: {lambert_first_arc_initial_state} [m , m/s]')
print(f'ECI state 0: {r_eci_array[0]} [m]')
print(f'ECI norm 0: {np.linalg.norm(r_eci_array[0])} [m], {np.linalg.norm(r_eci_array[0])/1000} [km]')
print(f'ECI state -1: {r_eci_array[jump_indices[0]]} [m]')
print(f'ECI norm -1: {np.linalg.norm(r_eci_array[jump_indices[0]])} [m], {np.linalg.norm(r_eci_array[jump_indices[0]])/1000} [km]')
print(f'pos norm: {np.linalg.norm(np.array(lambert_first_arc_initial_state)[:3])} [m], {np.linalg.norm(np.array(lambert_first_arc_initial_state)[:3])/1000} [km]')
print(f'vel norm: {np.linalg.norm(np.array(lambert_first_arc_initial_state)[3:])} [m/s], {np.linalg.norm(np.array(lambert_first_arc_initial_state)[3:])/1000} [km/s]')
print('')
print('Second lambert targetter')
print(f'Second lambert state 0: {lambert_second_arc_initial_state} [m, m/s]')
print(f'ECI state 0: {r_eci_array[jump_indices[0]+1]} [m]')
print(f'ECI norm 0: {np.linalg.norm(r_eci_array[jump_indices[0]+1])} [m]')
print(f'ECI state -1: {r_eci_array[jump_indices[1]]} [m]')
print(f'ECI norm -1: {np.linalg.norm(r_eci_array[jump_indices[1]])} [m]')
print(f'pos norm: {np.linalg.norm(np.array(lambert_second_arc_initial_state)[:3])} [m]')
print(f'vel norm: {np.linalg.norm(np.array(lambert_second_arc_initial_state)[3:])} [m/s]')
print('')
print('Third lambert targetter')
print(f'Third lambert state 0: {lambert_third_arc_initial_state} [m, m/s]')
print(f'ECI state 0: {r_eci_array[jump_indices[1]+1]} [m]')
print(f'ECI norm 0: {np.linalg.norm(r_eci_array[jump_indices[1]+1])} [m]')
print(f'ECI state -1: {r_eci_array[-1]} [m]')
print(f'ECI norm -1: {np.linalg.norm(r_eci_array[-1])} [m]')
print(f'pos norm: {np.linalg.norm(np.array(lambert_third_arc_initial_state)[:3])} [m]')
print(f'vel norm: {np.linalg.norm(np.array(lambert_third_arc_initial_state)[3:])} [m/s]')
print(f'''--------------------------------------------------------------------------------------------------------------''')
print(f'State at detected maneouvre: {manouvre_begin_state.flatten()} [m, m/s]')
print(f'detected maneouvre state pos norm: {np.linalg.norm(np.array(manouvre_begin_state)[:3])} [m]')
print(f'detected maneouvre state vel norm: {np.linalg.norm(np.array(manouvre_begin_state)[3:])} [m/s]')
print(f'''--------------------------------------------------------------------------------------------------------------''')
print(f'Original state X0: {X0.flatten()} [m, m/s]')
print(f'Original state pos norm: {np.linalg.norm(np.array(X0)[:3])} [m]')
print(f'Original state vel norm: {np.linalg.norm(np.array(X0)[3:])} [m/s]')
print(f'''--------------------------------------------------------------------------------------------------------------''')
print('Computing UKF for debugging Lambert....')
print(f'''--------------------------------------------------------------------------------------------------------------''')
state_params_debug, meas_dict_debug, sensor_params_debug = EstUtil.read_measurement_file(means_maneuver)
meas_dict_debug['tk_list'] = meas_dict_debug['tk_list']
meas_dict_debug['Yk_list'] = meas_dict_debug['Yk_list']

dt_list_debug = np.diff(meas_dict_debug['tk_list'])
jump_indices_debug = np.where(dt_list_debug > 10)[0]
fifth_jump_debug = jump_indices_debug[4] #==> manually via measurements
target_epoch_debug = meas_dict_debug['tk_list'][fifth_jump_debug + 1] 
target_index_debug = np.where(meas_dict_debug['tk_list'] == target_epoch_debug)[0][0]
filter_output_debug = EstUtil.ukf(state_params_debug, meas_dict_debug, sensor_params_debug, int_params, filter_params, bodies,cutoff=258)

time_steps_debug = list(filter_output_debug.keys())
for tk_debug in time_steps_debug:
    states_debug = filter_output_debug[tk_debug]['state']
print('results UKF:')
print(f'state at first radar measurment: {states_debug.flatten()}    [m, m/s]')
print(f'Position norm: {np.linalg.norm(states_debug.flatten()[:3])}  [m]')
print(f'velocity norm: {np.linalg.norm(states_debug.flatten()[3:])}  [m/s]')
print('')
print('Difference to Lambert')
print(f'Position norm: {(np.linalg.norm(np.array(lambert_first_arc_initial_state)[:3] - states_debug.flatten()[:3]))}  [m]')
print(f'velocity norm: {np.linalg.norm(np.array(lambert_first_arc_initial_state)[3:] - states_debug.flatten()[3:])}  [m/s]')
'''
LSQ method to estimate position and velocities between the sparse data. 
'''
print(f'''--------------------------------------------------------------------------------------------------------------''')
print("Performing LSQ to estimate positions and velocities of the new orbit....")
print(f'''--------------------------------------------------------------------------------------------------------------''')

#states at the beginning of the lambert arc
observed_state_begin = np.array([
    lambert_first_arc_initial_state,
    # lambert_second_arc_initial_state,
    # lambert_third_arc_initial_state
])

#states at the end of the lambert arc
observed_state_end = np.array([
    lambert_first_arc_final_state,
    # lambert_second_arc_final_state,
    # lambert_third_arc_final_state
])

#Epochs at the beginning of the lambert arc
epoch_b = np.array([
    tk_array[0], 
    # tk_array[jump_indices[0]+1],
    # tk_array[jump_indices[1]+1]
])

#Epochs at the end of the lambert arc
epoch_e = np.array([
    tk_array[jump_indices[0]], 
    # tk_array[jump_indices[1]],
    # tk_array[-1]
])

int_params = {}
int_params['tudat_integrator'] = 'rkf78'
int_params['step'] = 10
int_params['max_step'] = 1000.
int_params['min_step'] = 0.0001
int_params['rtol'] = 1e-12
int_params['atol'] = 1e-12

current_obj = rso_dict[91000]
rso_params = {
    'mass': current_obj['mass'],
    'area': current_obj['area'],
    'Cd': current_obj['Cd'],
    'Cr': current_obj['Cr'],
    'sph_deg': 8,
    'sph_ord': 8,
    'central_bodies': ['Earth'],
    'bodies_to_create': ['Earth', 'Sun', 'Moon']
}

#propagate the states
def propagate_state(initial_state, epoch0, target_epochs, rso_params, int_params):
    """Propagate from initial_state (6D) to each target_epoch."""
    results = []
    for epoch in target_epochs:
        t_vec = [epoch0, epoch]
        full_state_history = prop.propagate_orbit(initial_state, t_vec, rso_params, int_params)
        final_state = full_state_history[1][-1,:]
        results.append(final_state)
    return np.array(results).flatten()

#cost function for the LSQ
def cost_function(state_guess):
    init_state = np.array(state_guess)
    predicted = propagate_state(init_state, epoch_b[0], epoch_e, rso_params, int_params)
    residual = predicted - observed_state_end.flatten()
    weight = np.array([1e-3, 1e-3, 1e-3, 1, 1, 1])  # Adjust as needed
    return residual * weight


initial_guess = lambert_first_arc_initial_state.flatten()
bounds = (
    [-1e7, -1e7, -1e7, -15e3, -15e3, -15e3],
    [ 1e7,  1e7,  1e7,  15e3,  15e3,  15e3]
)

result = least_squares(
    cost_function,
    initial_guess,
    method='lm', #trf , lm (no bounds)
    # max_nfev=50,   #limits the number of iterations
    verbose=2,      #Progress of the LSQ 
    # bounds = bounds
)

# Final result
best_state = result.x
best_position = best_state[:3]
best_velocity = best_state[3:]

# J = result.jac
# cov = np.linalg.inv(J.T @ J)


print("Best-fit position:", best_position, 'm')
print("Best-fit velocity:", best_velocity, 'm/s')
print("Position norm:", np.linalg.norm(best_position), 'm')
print("Velocity norm:", np.linalg.norm(best_velocity), 'm/s')
# print("covariance", cov)
# print("standard deviation", np.array([np.sqrt(np.diag(cov))])  )
'''
Computing dV by finding the needed velocity to go to the last radar measurement via the lambert targetter,
annd substract initial vs needed velocity.
'''
print(f'''--------------------------------------------------------------------------------------------------------------''')
print("Computing required dV")
print(f'''--------------------------------------------------------------------------------------------------------------''')
initial_state,final_state = prop.lambert_targerter(
        manouvre_begin_state[:3].flatten(), best_state[:3], tk_array[jump_indices[0]], manoeuvre_epoch, bodies
    )
Velocity_needed = np.array(initial_state[3:]).flatten()
Velocity_currennt = np.array(manouvre_begin_state[3:]).flatten()
Change = Velocity_needed-Velocity_currennt
print(f'dV vector {Change} [m/s]')
print(f'dV nonrm {np.linalg.norm(Change)} [m/s]')
'''
Propagate object 91000 with TUDAT
save the data in propagated_data under the post mannouvre name
'''
print(f'''--------------------------------------------------------------------------------------------------------------''')
print("Propagating X0 before maneouvre and X0 after maneouvre for data visualization")
print(f'''--------------------------------------------------------------------------------------------------------------''')
t_obj, States_obj = prop.propagate_orbit(X0, t_vec, rso_params, int_params)

propagated_data['pre-manouvre'] = {
    'state': States_obj
    }

# propagated_data['manouvre-locationn'] = {
#     'state': np.array(manouvre_begin_state)
#     }

t_obj, States_obj = prop.propagate_orbit(best_state, t_vec, rso_params, int_params)
propagated_data['post-manouvre'] = {
    'state': States_obj
    }
'''
3D plot and animation to visualize whats going on
'''
def plot_3d_orbit_all(propagated_data):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    _u, _v = np.mgrid[0:2*np.pi:50j, 0:np.pi:40j] 
    _x = 6378000 * np.cos(_u) * np.sin(_v)
    _y = 6378000 * np.sin(_u) * np.sin(_v)
    _z = 6378000 * np.cos(_v)
    ax.plot_wireframe(_x, _y, _z, color="r", alpha=0.5, lw=0.5, zorder=0)

    for obj_id, data in propagated_data.items():
        state = data['state']
        x, y, z = state[:, 0], state[:, 1], state[:, 2]

        ax.plot(x, y, z, label=f'{obj_id}', linewidth=1.5)
        ax.scatter(x[0], y[0], z[0], color='green', s=30)
        ax.scatter(x[-1], y[-1], z[-1], color='red', s=30)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('3D Orbits of All Propagated Objects')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

def animate_3d_orbit_all(propagated_data, interval=50, save=False, filename='orbits.mp4'):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot Earth
    _u, _v = np.mgrid[0:2*np.pi:50j, 0:np.pi:40j] 
    _x = 6378000 * np.cos(_u) * np.sin(_v)
    _y = 6378000 * np.sin(_u) * np.sin(_v)
    _z = 6378000 * np.cos(_v)
    ax.plot_wireframe(_x, _y, _z, color="r", alpha=0.5, lw=0.5, zorder=0)

    # Get max length of trajectories
    max_len = max(len(data['state']) for data in propagated_data.values())

    # Store line objects
    orbit_lines = {}
    scatters = {}

    for obj_id, data in propagated_data.items():
        state = data['state']
        line, = ax.plot([], [], [], label=f'{obj_id}', lw=1.5)
        scatter = ax.scatter([], [], [], color='black', s=30)
        orbit_lines[obj_id] = (line, state)
        scatters[obj_id] = scatter

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_title('Animated 3D Orbits of All Propagated Objects')
    ax.legend()
    ax.grid(True)

    # Set fixed limits based on all data
    all_states = np.concatenate([data['state'] for data in propagated_data.values()])
    margin = 1.1 * np.max(np.abs(all_states))
    ax.set_xlim(-margin, margin)
    ax.set_ylim(-margin, margin)
    ax.set_zlim(-margin, margin)

    def update(frame):
        for obj_id, (line, state) in orbit_lines.items():
            if frame < len(state):
                x, y, z = state[:frame+1, 0], state[:frame+1, 1], state[:frame+1, 2]
                line.set_data(x, y)
                line.set_3d_properties(z)

                scatters[obj_id]._offsets3d = ([x[-1]], [y[-1]], [z[-1]])
        return [line for line, _ in orbit_lines.values()] + list(scatters.values())

    ani = FuncAnimation(fig, update, frames=max_len, interval=interval, blit=False)

    # if save:
    #     ani.save(filename, writer='ffmpeg')
    # else:
    plt.tight_layout()
    plt.show()

plot_3d_orbit_all(propagated_data)
# animate_3d_orbit_all(propagated_data)