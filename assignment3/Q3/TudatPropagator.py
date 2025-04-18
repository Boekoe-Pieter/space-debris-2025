import numpy as np


# Load required tudatpy modules
from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment
from tudatpy.numerical_simulation import environment_setup
from tudatpy.numerical_simulation import propagation_setup
from tudatpy.astro.time_conversion import DateTime
from tudatpy.astro import element_conversion
from tudatpy.util import result2array

# Load spice kernels
spice.load_standard_kernels()




def tudat_initialize_bodies(bodies_to_create=[]):
    '''
    This function initializes the bodies object for use in the Tudat 
    propagator. For the cases considered, only Earth, Sun, and Moon are needed,
    with Earth as the frame origin.
    
    Parameters
    ------
    bodies_to_create : list, optional (default=[])
        list of bodies to create, if empty, will use default Earth, Sun, Moon
    
    Returns
    ------
    bodies : tudat object
    
    '''

    # Define string names for bodies to be created from default.
    if len(bodies_to_create) == 0:
        bodies_to_create = ["Sun", "Earth", "Moon"]

    # Use "Earth"/"J2000" as global frame origin and orientation.
    global_frame_origin = "Earth"
    global_frame_orientation = "J2000"

    # Create default body settings, usually from `spice`.
    body_settings = environment_setup.get_default_body_settings(
        bodies_to_create,
        global_frame_origin,
        global_frame_orientation)

    # Create system of selected celestial bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)
    
    return bodies


def propagate_orbit(Xo, tvec, state_params, int_params, bodies=None):
    '''
    This function propagates an orbit using physical parameters provided in 
    state_params and integration parameters provided in int_params.
    
    Parameters
    ------
    Xo : 6x1 numpy array
        Cartesian state vector [m, m/s]
    tvec : list or numpy array
        propagator will only use first and last terms to set the initial and
        final time of the propagation, intermediate times are ignored
        
        [t0, ..., tf] given as time in seconds since J2000
        
    state_params : dictionary
        propagator parameters
        
        fields:
            Cd: float, drag coefficient
            Cr: float, reflectivity coefficient
            area: float [m^2]
            mass: float [kg]
            sph_deg: int, spherical harmonics expansion degree for Earth
            sph_ord: int, spherical harmonics expansion order for Earth
            central_bodies: list of central bodies for propagator ["Earth"]
            bodies_to_create: list of bodies to create ["Earth", "Sun", "Moon"]
            
    int_params : dictionary
        numerical integration parameters
        
    bodies : tudat object, optional
        contains parameters for the environment bodies used in propagation
        if None, will initialize with bodies given in state_params
        
    Returns
    ------
    tout : N element numpy array
        times of propagation output in seconds since J2000
    Xout : Nxn numpy array
        each row Xout[k,:] corresponds to Cartesian state at time tout[k]        
    
    '''
    
    # Initial state
    initial_state = Xo.flatten()
    
    # Retrieve input parameters
    central_bodies = state_params['central_bodies']
    bodies_to_create = state_params['bodies_to_create']
    mass = state_params['mass']
    Cd = state_params['Cd']
    Cr = state_params['Cr']
    area = state_params['area']
    sph_deg = state_params['sph_deg']
    sph_ord = state_params['sph_ord']
    
    # Simulation start and end
    simulation_start_epoch = tvec[0]
    simulation_end_epoch = tvec[-1]
    
    # Setup bodies
    if bodies is None:
        bodies = tudat_initialize_bodies(bodies_to_create)
    
    
    # Create the bodies to propagate
    # TUDAT always uses 6 element state vector
    N = int(len(Xo)/6)
    central_bodies = central_bodies*N
    bodies_to_propagate = []
    for jj in range(N):
        jj_str = str(jj)
        bodies.create_empty_body(jj_str)
        bodies.get(jj_str).mass = mass
        bodies_to_propagate.append(jj_str)
        
        if Cd > 0.:
            aero_coefficient_settings = environment_setup.aerodynamic_coefficients.constant(
                area, [Cd, 0, 0]
            )
            environment_setup.add_aerodynamic_coefficient_interface(
                bodies, jj_str, aero_coefficient_settings)
            
        if Cr > 0.:
            # occulting_bodies = ['Earth']
            # radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
            #     'Sun', srp_area_m2, Cr, occulting_bodies
            # )
            # environment_setup.add_radiation_pressure_interface(
            #     bodies, jj_str, radiation_pressure_settings)
            
            occulting_bodies_dict = dict()
            occulting_bodies_dict[ "Sun" ] = [ "Earth" ]
            
            radiation_pressure_settings = environment_setup.radiation_pressure.cannonball_radiation_target(
                area, Cr, occulting_bodies_dict )
            
            environment_setup.add_radiation_pressure_target_model(
                bodies, jj_str, radiation_pressure_settings)
            

    acceleration_settings_setup = {}        
    if 'Earth' in bodies_to_create:
        
        # Gravity
        if sph_deg == 0 and sph_ord == 0:
            acceleration_settings_setup['Earth'] = [propagation_setup.acceleration.point_mass_gravity()]
        else:
            acceleration_settings_setup['Earth'] = [propagation_setup.acceleration.spherical_harmonic_gravity(sph_deg, sph_ord)]
        
        # Aerodynamic Drag
        if Cd > 0.:                
            acceleration_settings_setup['Earth'].append(propagation_setup.acceleration.aerodynamic())
        
    if 'Sun' in bodies_to_create:
        
        # Gravity
        acceleration_settings_setup['Sun'] = [propagation_setup.acceleration.point_mass_gravity()]
        
        # Solar Radiation Pressure
        if Cr > 0.:                
            #acceleration_settings_setup['Sun'].append(propagation_setup.acceleration.cannonball_radiation_pressure())
            acceleration_settings_setup['Sun'].append(propagation_setup.acceleration.radiation_pressure())
    
    if 'Moon' in bodies_to_create:
        
        # Gravity
        acceleration_settings_setup['Moon'] = [propagation_setup.acceleration.point_mass_gravity()]
    

    acceleration_settings = {}
    for jj in range(N):
        acceleration_settings[str(jj)] = acceleration_settings_setup
        
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies
    )
    

    # Create termination settings
    termination_condition = propagation_setup.propagator.time_termination(
        simulation_end_epoch, terminate_exactly_on_final_condition=True
    )


    # Create numerical integrator settings
    if int_params['tudat_integrator'] == 'rk4':
        fixed_step_size = int_params['step']
        integrator_settings = propagation_setup.integrator.runge_kutta_4(
            fixed_step_size
        )
        
    elif int_params['tudat_integrator'] == 'rkf78':
        initial_step_size = int_params['step']
        maximum_step_size = int_params['max_step']
        minimum_step_size = int_params['min_step']
        rtol = int_params['rtol']
        atol = int_params['atol']
        integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
            initial_step_size,
            propagation_setup.integrator.CoefficientSets.rkf_78,
            minimum_step_size,
            maximum_step_size,
            rtol,
            atol)
    
        
        
    # Create propagation settings
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        simulation_start_epoch,
        integrator_settings,
        termination_condition
    )
    
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies, propagator_settings )

    # Extract the resulting state history and convert it to an ndarray
    states = dynamics_simulator.state_history
    states_array = result2array(states)        
    
    
    tout = states_array[:,0]
    Xout = states_array[:,1:6*N+1]
    
    
    return tout, Xout


def propagate_state_and_covar(Xo, Po, tvec, state_params, int_params, bodies=None, alpha=1e-4):
    
    # Sigma point weights 
    n = len(Xo)
    beta = 2.
    kappa = 3. - n
    lam = alpha**2.*(n + kappa) - n
    gam = np.sqrt(n + lam)
    Wm = 1./(2.*(n + lam)) * np.ones(2*n,)
    Wc = Wm.copy()
    Wm = np.insert(Wm, 0, lam/(n + lam))
    Wc = np.insert(Wc, 0, lam/(n + lam) + (1 - alpha**2 + beta))
    diagWc = np.diag(Wc)
    
    # Sigma point generation 
    sqP = np.linalg.cholesky(Po)
    Xrep = np.tile(Xo, (1, n))
    chi = np.concatenate((Xo, Xrep+(gam*sqP), Xrep-(gam*sqP)), axis=1)
    
    # Parameter extraction 
    dyn_params = state_params.copy()  
    phys_vars = chi[6:8,:] 
    area_const = dyn_params['area']  
    mass_const = dyn_params['mass'] 
    
    # Initialize storage for propagated points 
    propagated_states = np.zeros((n, 2*n+1))  
    
    for point_idx in range(2*n+1): 
        # Current sigma point 
        curr_point = chi[:, point_idx]  
        pos_vel = curr_point[:6] 
        drag_coeff = curr_point[6]  
        reflect_coeff = curr_point[7] 
        
        # Body setup 
        if bodies is None:
            env_bodies = tudat_initialize_bodies(dyn_params['bodies_to_create'])
        else:
            env_bodies = bodies
            
        # Create unique body for this propagation 
        body = f"prop_body_{point_idx}"  
        env_bodies.create_empty_body(body)
        env_bodies.get(body).mass = mass_const
        
        # Aerodynamic setup 
        if drag_coeff > 0.:
            drag_config = environment_setup.aerodynamic_coefficients.constant(
                area_const, [drag_coeff, 0, 0]  
            )
            environment_setup.add_aerodynamic_coefficient_interface(
                env_bodies, body, drag_config)
            
        # Radiation pressure 
        if reflect_coeff > 0.:
            shadow_bodies = {"Sun": ["Earth"]}  
            srp_config = environment_setup.radiation_pressure.cannonball_radiation_target(
                area_const, reflect_coeff, shadow_bodies  
            )
            environment_setup.add_radiation_pressure_target_model(
                env_bodies, body, srp_config)
        
        # Acceleration setup 
        accel_models = {}
        accel_models[body] = {}
        
        if 'Earth' in dyn_params['bodies_to_create']:
            if dyn_params['sph_deg'] == 0 and dyn_params['sph_ord'] == 0:
                accel_models[body]['Earth'] = [
                    propagation_setup.acceleration.point_mass_gravity()
                ]
            else:
                accel_models[body]['Earth'] = [
                    propagation_setup.acceleration.spherical_harmonic_gravity(
                        dyn_params['sph_deg'], dyn_params['sph_ord'])
                ]
            if drag_coeff > 0.:
                accel_models[body]['Earth'].append(
                    propagation_setup.acceleration.aerodynamic()
                )
                
        if 'Sun' in dyn_params['bodies_to_create']:
            accel_models[body]['Sun'] = [
                propagation_setup.acceleration.point_mass_gravity()
            ]
            if reflect_coeff > 0.:
                accel_models[body]['Sun'].append(
                    propagation_setup.acceleration.radiation_pressure()
                )
                
        if 'Moon' in dyn_params['bodies_to_create']:
            accel_models[body]['Moon'] = [
                propagation_setup.acceleration.point_mass_gravity()
            ]
            
        # Create models 
        acceleration_models = propagation_setup.create_acceleration_models(
            env_bodies, accel_models, [body], ["Earth"])
            
        # Integrator setup 
        if int_params['tudat_integrator'] == 'rk4':
            integrator = propagation_setup.integrator.runge_kutta_4(
                int_params['step'])
        elif int_params['tudat_integrator'] == 'rkf78':
            integrator = propagation_setup.integrator.runge_kutta_variable_step_size(
                int_params['step'],
                propagation_setup.integrator.CoefficientSets.rkf_78,
                int_params['min_step'],
                int_params['max_step'],
                int_params['rtol'],
                int_params['atol'])
                
        # Propagation 
        stop_condition = propagation_setup.propagator.time_termination(
            tvec[-1], terminate_exactly_on_final_condition=True)
            
        setup = propagation_setup.propagator.translational(
            ["Earth"],
            acceleration_models,
            [body],
            pos_vel,
            tvec[0],
            integrator,
            stop_condition)
            
        dynamics_results = numerical_simulation.create_dynamics_simulator(
            env_bodies, setup)
        final_states = result2array(dynamics_results.state_history)
        
        propagated_states[:, point_idx] = np.concatenate([
            final_states[-1, 1:7],  # Position/velocity
            [drag_coeff, reflect_coeff]  # Estimated parameters
        ])
    
    # Compute final state and covariance 
    Xf = np.dot(propagated_states, Wm.reshape(-1, 1))
    diff = propagated_states - np.dot(Xf, np.ones((1, 2*n+1)))
    Pf = np.dot(diff, np.dot(diagWc, diff.T))
    
    return tvec[-1], Xf, Pf