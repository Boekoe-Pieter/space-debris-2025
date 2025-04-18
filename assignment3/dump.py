
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
        state_positions = full_state_history[1][:,:3]
        results.append(final_state)
    return np.array(results).flatten()

#cost function for the LSQ
def cost_function(state_guess):
    init_state = np.array(state_guess)
    predicted = propagate_state(init_state, epoch_b[0], epoch_e, rso_params, int_params)
    residual = predicted - observed_state_end.flatten()
    # weight = np.array([1e-3, 1e-3, 1e-3, 1, 1, 1])  # Adjust as needed
    return residual #* weight

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