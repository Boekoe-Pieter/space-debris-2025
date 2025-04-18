import numpy as np
from scipy.optimize import least_squares
from tudatpy.astro import element_conversion
import EstimationUtilities as EstUtil
import TudatPropagator as prop
import ConjunctionUtilities as ConjUtil
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load data
meas_file = r"C:\Users\kiira\OneDrive\Desktop\Aerospace\Q3\Space Debris\Assignment 3\space-debris-2025\assignment3\data\group1\q4_meas_iod_99001.pkl"
state_params, meas_dict, sensor = EstUtil.read_measurement_file(meas_file)

tk_list = meas_dict['tk_list']
Yk_list = meas_dict['Yk_list']

if len(tk_list) < 2:
    raise ValueError("Need at least two measurements")

bodies = prop.tudat_initialize_bodies(['Earth', 'Sun', 'Moon'])

state_params = {'mass': 100.0, 'area': 1.0, 'Cd': 2.2, 'Cr': 1.3, 'sph_deg': 8, 'sph_ord': 8, 'central_bodies': ['Earth'], 'bodies_to_create': ['Earth', 'Sun', 'Moon']}
int_params = {'tudat_integrator': 'rkf78', 'step': 10.0, 'max_step': 1000.0, 'min_step': 1e-3, 'rtol': 1e-6, 'atol': 1e-6}

def compute_r_eci(Yk, sensor, tk, bodies):
    sensor_ecef = sensor['sensor_ecef']
    rot = bodies.get("Earth").rotation_model.body_fixed_to_inertial_rotation(tk)
    sensor_eci = np.dot(rot, sensor_ecef).flatten()
    rg, ra, dec = Yk.flatten()
    rho_hat = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    return sensor_eci + rg * rho_hat

t0, tk2 = tk_list[0], tk_list[1]
Yk1, Yk2 = Yk_list[0], Yk_list[1]
r1 = compute_r_eci(Yk1, sensor, t0, bodies)
r2 = compute_r_eci(Yk2, sensor, tk2, bodies)

print("r1:", r1)
print("r2:", r2)
print("dt:", tk2 - t0)

def lambert_solver(r1, r2, dt, mu):
    r1, r2 = r1.flatten(), r2.flatten()
    r1_norm, r2_norm = np.linalg.norm(r1), np.linalg.norm(r2)
    cos_dnu = np.dot(r1, r2) / (r1_norm * r2_norm)
    sin_dnu = np.sign(np.cross(r1, r2)[-1]) * np.sqrt(1 - cos_dnu**2)
    A = sin_dnu * np.sqrt(r1_norm * r2_norm / (1 - cos_dnu))
    z = 1.0
    for _ in range(50):
        C = (1 - np.cos(np.sqrt(z))) / z if z > 1e-6 else 0.5 - z / 24
        S = (np.sqrt(z) - np.sin(np.sqrt(z))) / (np.sqrt(z)**3) if z > 1e-6 else 1/6 - z / 120
        y = r1_norm + r2_norm + A * (z * S - 1) / np.sqrt(C)
        F = (y / C)**1.5 * S + A * np.sqrt(y) - np.sqrt(mu) * dt
        dFdz = (y / C)**1.5 * (1 / (2 * z) * (C - 3 * S / (2 * C)) + 3 * S**2 / (4 * C)) + A / 8 * (3 * S / C * np.sqrt(y) + A * np.sqrt(C / y))
        dz = F / dFdz if abs(dFdz) > 1e-10 else 0
        z -= dz
        if abs(dz) < 1e-8:
            break
    if z < 0 or np.isnan(z):
        return (r2 - r1) / dt
    chi = np.sqrt(z)
    alpha = 1 - z * r1_norm / mu
    v1 = (r2 - r1 * (1 - chi**2 * C / alpha)) / (np.sqrt(mu) * dt * np.sqrt(C) / chi)
    if np.any(np.isnan(v1)) or np.any(np.isinf(v1)):
        return (r2 - r1) / dt
    return v1

mu = bodies.get("Earth").gravitational_parameter
dt = tk2 - t0
v1 = lambert_solver(r1, r2, dt, mu)
X0_guess = np.hstack((r1, v1))

print("X0_guess:", X0_guess)

def residuals(X0, t0, meas_dict, sensor, state_params, int_params, bodies):
    resids = []
    for tk, Yk in zip(meas_dict['tk_list'], meas_dict['Yk_list']):
        if tk == t0:
            Xk = X0.reshape(6, 1)
        else:
            tvec = [t0, tk]
            tout, Xout = prop.propagate_orbit(X0, tvec, state_params, int_params, bodies)
            Xk = Xout[-1, :].reshape(6, 1)
        Y_pred = EstUtil.compute_measurement(tk, Xk, sensor, bodies)
        res = (Yk - Y_pred).flatten()
        for j, mtype in enumerate(sensor['meas_types']):
            res[j] /= sensor['sigma_dict'][mtype]
        resids.extend(res)
    return np.array(resids)

result = least_squares(residuals, X0_guess, args=(t0, meas_dict, sensor, state_params, int_params, bodies), method='lm')
X0_est = result.x.reshape(6, 1)
J = result.jac

sigma_list = [sensor['sigma_dict'][mtype] for mtype in sensor['meas_types']]
W = np.diag([1 / sigma**2 for sigma in sigma_list for _ in range(len(tk_list))])
P0 = np.linalg.inv(J.T @ W @ J + 1e-6 * np.eye(6))

print("Eigenvalues of P0:", np.linalg.eigvals(P0))

elements = element_conversion.cartesian_to_keplerian(X0_est, mu)
print("Q4(a) Results:")
print("Keplerian Elements:", elements)
print("State at t0:", X0_est.flatten())
print("Covariance:", P0)

R_earth = 6371e3
perigee = elements[0] * (1 - elements[1])
if perigee < R_earth:
    print("Perigee below Earth surface")

print("\nQ4(b) Results:")
print("State at t0:", X0_est.flatten())
print("Covariance:", P0)

rso_file = r"C:\Users\kiira\OneDrive\Desktop\Aerospace\Q3\Space Debris\Assignment 3\space-debris-2025\assignment3\data\group1\estimated_rso_catalog.pkl"
rso_dict = ConjUtil.read_catalog_file(rso_file)
oo_id = list(rso_dict.keys())[0]
oo_params = rso_dict[oo_id].copy()
oo_state = oo_params['state']
oo_params.update({'central_bodies': ['Earth'], 'bodies_to_create': ['Earth', 'Sun', 'Moon'], 'sph_deg': 8, 'sph_ord': 8})

oo_elements = element_conversion.cartesian_to_keplerian(oo_state, mu)
perigee_new = elements[0] * (1 - elements[1])
perigee_oo = oo_elements[0] * (1 - oo_elements[1])
print("Hazard Assessment:", "Potential hazard" if abs(perigee_new - perigee_oo) < 100e3 else "No hazard")

trange = [t0, t0 + 48 * 3600]
P0_inflated = P0 * 2
print("Inflated Covariance:", P0_inflated)

T_list, rho_list = ConjUtil.compute_TCA(X0_est, oo_state, trange, state_params, oo_params, int_params, bodies)
if not T_list:
    print("No conjunction in 48 hours")
else:
    tca, rho_min = T_list[0], rho_list[0]
    print(f"TCA at {tca} s, Distance = {rho_min} m")

    tout_rso, Xout_rso = prop.propagate_orbit(X0_est, [t0, tca], state_params, int_params, bodies)
    X_rso_tca = Xout_rso[-1]
    tout_oo, Xout_oo = prop.propagate_orbit(oo_state, [t0, tca], oo_params, int_params, bodies)
    X_oo_tca = Xout_oo[-1]

    Phi = np.zeros((6, 6))
    for i in range(6):
        X0_pert = X0_est.copy()
        X0_pert[i] += 1e-3
        _, Xout_pert = prop.propagate_orbit(X0_pert, [t0, tca], state_params, int_params, bodies)
        Phi[:, i] = (Xout_pert[-1] - X_rso_tca) / 1e-3
    print("STM at TCA:", Phi)

    P_rso_tca = Phi @ P0_inflated @ Phi.T
    print("Covariance at TCA:", P_rso_tca)

    P_oo_tca = rso_dict[oo_id]['covar']
    rel_vel = np.linalg.norm(X_rso_tca[3:] - X_oo_tca[3:])
    Pc = ConjUtil.Pc2D_Foster(X_rso_tca, P_rso_tca, X_oo_tca, P_oo_tca, 10.0)

    tca_datetime = datetime(2000, 1, 1, 12, 0, 0) + timedelta(seconds=tca)
    print("\nQ4(c) Results:")
    print(f"TCA: {tca_datetime.isoformat()}Z")
    print(f"Distance: {rho_min} m")
    print(f"Relative Velocity: {rel_vel} m/s")
    print(f"Collision Probability: {Pc}")
    print("Decision:", "Maneuver recommended" if Pc > 1e-4 or rho_min < 1000 else "No maneuver")

def plot_distance(X0_est, oo_state, trange, state_params, int_params, bodies, tca, rho_min):
    tvec = np.linspace(trange[0], trange[1], 1000)
    tout_rso, Xout_rso = prop.propagate_orbit(X0_est, tvec, state_params, int_params, bodies)
    tout_oo, Xout_oo = prop.propagate_orbit(oo_state, tvec, state_params, int_params, bodies)
    X_rso = np.column_stack([np.interp(tvec, tout_rso, Xout_rso[:, i]) for i in range(3)])
    X_oo = np.column_stack([np.interp(tvec, tout_oo, Xout_oo[:, i]) for i in range(3)])
    distances = np.linalg.norm(X_rso - X_oo, axis=1) / 1000
    times = (tvec - trange[0]) / 3600
    plt.figure(figsize=(10, 6))
    plt.plot(times, distances, 'b-', label='Distance')
    plt.scatter([(tca - trange[0]) / 3600], [rho_min / 1000], c='r', s=100, label='TCA')
    plt.xlabel('Time (hours)')
    plt.ylabel('Distance (km)')
    plt.title('Distance between RSO and O/O')
    plt.legend()
    plt.grid()
    plt.savefig('distance.pdf')
    plt.close()

# def plot_orbits(X0_est, oo_state, trange, state_params, int_params, bodies, tca=None):
#     tvec = np.linspace(trange[0], trange[1], 1000)
#     tout_rso, Xout_rso = prop.propagate_orbit(X0_est, tvec, state_params, int_params, bodies)
#     tout_oo, Xout_oo = prop.propagate_orbit(oo_state, tvec, state_params, int_params, bodies)
#     Xout_rso_km = Xout_rso / 1000
#     Xout_oo_km = Xout_oo / 1000
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot(Xout_rso_km[:, 0], Xout_rso_km[:, 1], Xout_rso_km[:, 2], 'r-', label='RSO')
#     ax.plot(Xout_oo_km[:, 0], Xout_oo_km[:, 1], Xout_oo_km[:, 2], 'b-', label='O/O')
#     if tca:
#         idx_rso = np.argmin(np.abs(tout_rso - tca))
#         idx_oo = np.argmin(np.abs(tout_oo - tca))
#         ax.scatter([Xout_rso_km[idx_rso, 0]], [Xout_rso_km[idx_rso, 1]], [Xout_rso_km[idx_rso, 2]], c='k', s=100, label='RSO at TCA')
#         ax.scatter([Xout_oo_km[idx_oo, 0]], [Xout_oo_km[idx_oo, 1]], [Xout_oo_km[idx_oo, 2]], c='k', s=100, label='O/O at TCA')
#     ax.set_xlabel('X (km)')
#     ax.set_ylabel('Y (km)')
#     ax.set_zlabel('Z (km)')
#     ax.legend()
#     plt.savefig('orbits.pdf')
#     plt.close()

if T_list:
    # plot_orbits(X0_est, oo_state, trange, state_params, int_params, bodies, tca)
    plot_distance(X0_est, oo_state, trange, state_params, int_params, bodies, tca, rho_min)