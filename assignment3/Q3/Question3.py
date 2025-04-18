import os
import numpy as np
import EstimationUtilities as EstUtil
import ConjunctionUtilities as ConjUtil
import TudatPropagator as prop
import matplotlib.pyplot as plt
import pickle
from matplotlib.ticker import AutoMinorLocator

# ---------------------------
# DATA LOADING CONFIGURATION
# ---------------------------
DATA_DIR = "assignment3/data/group1"
OPTICAL_PATH = os.path.join(DATA_DIR, "q3_optical_meas_objchar_91452.pkl")
RADAR_PATH = os.path.join(DATA_DIR, "q3_radar_meas_objchar_91452.pkl")

# Load measurement datasets
optical_params, optical_meas, optical_config = EstUtil.read_measurement_file(OPTICAL_PATH)
radar_params, radar_meas, radar_config = EstUtil.read_measurement_file(RADAR_PATH)

# --------------------------
# MEASUREMENT PREPROCESSING
# --------------------------
# Reshape measurement vectors
for sensor in [optical_meas, radar_meas]:
    sensor["Yk_list"] = [m.reshape(-1, 1) for m in sensor["Yk_list"]]

# Merge sensor datasets
merged_timestamps = np.concatenate([radar_meas["tk_list"], optical_meas["tk_list"]])
merged_measurements = radar_meas["Yk_list"] + optical_meas["Yk_list"]
sensor_tags = ["radar"] * len(radar_meas["tk_list"]) + ["optical"] * len(optical_meas["tk_list"])

# Temporal sorting
chronological_order = np.argsort(merged_timestamps)
organized_data = {
    "tk_list": merged_timestamps[chronological_order],
    "Yk_list": [merged_measurements[i] for i in chronological_order],
    "sensor_list": [sensor_tags[i] for i in chronological_order]
}

sensor_configurations = {
    "radar": radar_config,
    "optical": optical_config
}

# ----------------------------
# ESTIMATION INITIALIZATION
# ----------------------------
# Initial state vector and covariance
initial_state = np.vstack((
    radar_params["state"],
    np.array([radar_params["Cd"]]),
    np.array([radar_params["Cr"]])
))

dynamics_cov = radar_params["covar"]
parameter_cov = np.diag([5e-2, 1e-5])
initial_covariance = np.block([
    [dynamics_cov, np.zeros((6, 2))],
    [np.zeros((2, 6)), parameter_cov]
])

estimation_parameters = radar_params.copy()
estimation_parameters.update({
    "state": initial_state,
    "covar": initial_covariance
})

# Filter configuration
ukf_settings = {
    "Qeci": 1e-12 * np.eye(3),
    "Qric": 1e-12 * np.eye(3),
    "Qparam": 1e-10 * np.eye(2),
    "alpha": 1e-2,
    "gap_seconds": 400.0
}

integration_settings = {
    "tudat_integrator": "rkf78",
    "step": 2.0,
    "max_step": 20.0,
    "min_step": 1e-2,
    "rtol": 1e-7,
    "atol": 1e-7
}

# Initialize celestial system
celestial_bodies = prop.tudat_initialize_bodies(["Earth", "Sun", "Moon"])

# ---------------------
# FILTER EXECUTION
# ---------------------
result = EstUtil.ukf(
    estimation_parameters,
    organized_data,
    sensor_configurations,
    integration_settings,
    ukf_settings,
    celestial_bodies
)

# -------------------------
# DIAGNOSTICS & OUTPUT
# -------------------------
# Initial parameters report
print("INITIAL PARAMETER VALUES:")
print(f"Drag Coefficient (C_D): {radar_params['Cd']:.4f}")
print(f"Radiation Coefficient (C_R): {radar_params['Cr']:.4f}")
print(f"Mass: {radar_params['mass']:.1f} kg")
print(f"Cross-sectional Area: {radar_params['area']:.2f} m²")

# Measurement statistics
optical_count = sum(1 for s in organized_data["sensor_list"] if s == "optical")
radar_count = sum(1 for s in organized_data["sensor_list"] if s == "radar")
print(f"\nMEASUREMENT COUNT:\nOptical: {optical_count}\nRadar: {radar_count}")

# Final parameter estimates
final_estimate = result[organized_data["tk_list"][-1]]["state"]
final_uncertainty = result[organized_data["tk_list"][-1]]["covar"]

print("\nFINAL PARAMETER ESTIMATES:")
print(f"C_D = {final_estimate[6, 0]:.4f} ± {np.sqrt(final_uncertainty[6, 6]):.4f}")
print(f"C_R = {final_estimate[7, 0]:.4f} ± {np.sqrt(final_uncertainty[7, 7]):.4f}")

print("\nFINAL STATE VECTOR:")
position_labels = ['X', 'Y', 'Z']
velocity_labels = ['Vx', 'Vy', 'Vz']
for i, label in enumerate(position_labels):
    print(f"{label}: {final_estimate[i, 0]:8.1f} ± {np.sqrt(final_uncertainty[i, i]):5.1f} m")

for i, label in enumerate(velocity_labels):
    print(f"{label}: {final_estimate[i+3, 0]:10.6f} ± {np.sqrt(final_uncertainty[i+3, i+3]):8.6f} m/s")

# --------------------
# DATA VISUALIZATION
# --------------------
# Configure plot style

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 300,
    'savefig.bbox': 'tight'
})

def configure_axes(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.tick_params(which='major', length=6)
    ax.tick_params(which='minor', length=3)

# Save processed data
os.makedirs("results", exist_ok=True)
with open("results/visualization_data.pkl", "wb") as file:
    pickle.dump({
        "timestamps": organized_data["tk_list"],
        "sensor_types": organized_data["sensor_list"],
        "measurements": organized_data["Yk_list"],
        "estimates": result
    }, file)

# Load stored data for plotting
with open("results/visualization_data.pkl", "rb") as file:
    plot_data = pickle.load(file)

reference_time = plot_data["timestamps"][0]

# Organize measurement data
optical_times, optical_mag = [], []
radar_times, radar_range, radar_ra, radar_dec = [], [], [], []

for idx, timestamp in enumerate(plot_data["timestamps"]):
    elapsed_time = (timestamp - reference_time) / 3600
    measurement = plot_data["measurements"][idx].flatten()
    
    if plot_data["sensor_types"][idx] == "optical":
        optical_times.append(elapsed_time)
        optical_mag.append(measurement[0])
    else:
        radar_times.append(elapsed_time)
        radar_range.append(measurement[0])
        radar_ra.append(np.rad2deg(measurement[1]))
        radar_dec.append(np.rad2deg(measurement[2]))

# Generate optical measurement plot
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(optical_times, optical_mag, 'o', ms=2.5, mec='#2ca02c', mfc='none', mew=0.5)
ax.set(xlabel="Elapsed Time (hours)", ylabel="Apparent Magnitude",
       title="Optical Brightness Measurements")
configure_axes(ax)
plt.savefig("results/optical_measurements.png")
plt.close()

# Generate radar measurement plot
fig, host = plt.subplots(figsize=(9, 5))
par1 = host.twinx()
par2 = host.twinx()

host.plot(radar_times, radar_range, '^', ms=3, color='#1f77b4', alpha=0.7, label="Range")
par1.plot(radar_times, radar_ra, 's', ms=2.5, color='#ff7f0e', alpha=0.7, label="RA")
par2.plot(radar_times, radar_dec, 'D', ms=2, color='#2ca02c', alpha=0.7, label="DEC")

host.set_xlabel("Elapsed Time (hours)")
host.set_ylabel("Range (m)", color='#1f77b4')
par1.set_ylabel("Right Ascension (°)", color='#ff7f0e')
par2.set_ylabel("Declination (°)", color='#2ca02c')

par2.spines["right"].set_position(("axes", 1.15))
for ax in [host, par1, par2]:
    configure_axes(ax)
    ax.tick_params(axis='y')

fig.suptitle("Radar Tracking Measurements")
fig.tight_layout()
plt.savefig("results/radar_measurements.png")
plt.close()

# Residual analysis
optical_residuals, radar_residuals = [], []
opt_res_times, rad_res_times = [], []
radar_range_res, radar_ra_res, radar_dec_res = [], [], []

for idx, timestamp in enumerate(plot_data["timestamps"]):
    residual = plot_data["estimates"][timestamp]["resids"].flatten()
    elapsed = (timestamp - reference_time) / 3600
    
    if plot_data["sensor_types"][idx] == "optical":
        optical_residuals.append(residual[0])
        opt_res_times.append(elapsed)
    else:
        rad_res_times.append(elapsed)
        radar_range_res.append(residual[0] if len(residual) > 0 else np.nan)
        radar_ra_res.append(np.rad2deg(residual[1] if len(residual) > 1 else np.nan))
        radar_dec_res.append(np.rad2deg(residual[2] if len(residual) > 2 else np.nan))

# Optical residual plot
fig, ax = plt.subplots(figsize=(8, 4.5))
mu, sigma = np.mean(optical_residuals), np.std(optical_residuals)
ax.plot(opt_res_times, optical_residuals, 'o', ms=3, mec='#2ca02c', mfc='none', mew=0.5)
ax.axhline(mu, ls='--', c='k', lw=1, label=f'Mean: {mu:.2f}')
ax.fill_between(opt_res_times, mu-sigma, mu+sigma, color='#2ca02c', alpha=0.1, 
                label=f'±1σ ({sigma:.2f})')

ax.set(xlabel="Elapsed Time (hours)", ylabel="Residual Magnitude",
       title="Optical Measurement Residuals")
configure_axes(ax)
ax.legend()
plt.savefig("results/optical_residuals.png")
plt.close()

# Radar residual plot
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

def plot_radar_residuals(ax, data, color, ylabel):
    mu, sigma = np.nanmean(data), np.nanstd(data)
    ax.plot(rad_res_times, data, 'o', ms=2, mec=color, mfc='none', mew=0.5)
    ax.axhline(mu, ls='--', c='k', lw=1)
    ax.fill_between(rad_res_times, mu-sigma, mu+sigma, color=color, alpha=0.1)
    ax.set_ylabel(ylabel, color=color)
    configure_axes(ax)

plot_radar_residuals(ax1, radar_range_res, '#1f77b4', "Range Residual (m)")
plot_radar_residuals(ax2, radar_ra_res, '#ff7f0e', "RA Residual (°)")
plot_radar_residuals(ax3, radar_dec_res, '#2ca02c', "DEC Residual (°)")

ax3.set_xlabel("Elapsed Time (hours)")
fig.suptitle("Radar Measurement Residuals")
plt.tight_layout()
plt.savefig("results/radar_residuals.png")
plt.close()

# Calculate residual statistics
def rms(values):
    return np.sqrt(np.nanmean(np.square(values)))

print("\nPOST-FIT RESIDUAL ANALYSIS:")
print(f"Optical Magnitude RMS: {rms(optical_residuals):.4f}")
print(f"Radar Range RMS:       {rms(radar_range_res):.4f} m")
print(f"Radar RA RMS:          {rms(radar_ra_res):.6f}°")
print(f"Radar DEC RMS:         {rms(radar_dec_res):.6f}°")