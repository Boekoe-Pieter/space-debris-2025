import pickle
import numpy as np
import ConjunctionUtilities as cu
from tudatpy.astro.element_conversion import cartesian_to_keplerian, true_to_eccentric_anomaly, true_to_mean_anomaly, delta_mean_anomaly_to_elapsed_time

mu = 3.986004415e14 # [m^3/s^2]

def calculate_rp_ra(ae_arr:np.ndarray) -> np.ndarray[(2,)]:
    a, e = ae_arr

    return np.array([a * (1 - e), a * (1 + e)])

def perigee_apogee_filter(asset:dict, rso:dict, D:float, mu:float=mu) -> dict:
    asset_rp, asset_ra = calculate_rp_ra(cartesian_to_keplerian(asset["state"], mu)[:2])

    rso_keys = list(rso.keys())

    rso_rp_ra = np.array([calculate_rp_ra(cartesian_to_keplerian(object["state"], mu)[:2]) for object in rso.values()])

    q = np.where(rso_rp_ra[:,0] < asset_rp, asset_rp, rso_rp_ra[:,0])
    Q = np.where(rso_rp_ra[:,1] > asset_ra, asset_ra, rso_rp_ra[:,1])

    mask = q - Q > D

    filtered_rso = {}

    for i in range(q.shape[0]):
        if not mask[i]:
            current_key = rso_keys[i]
            filtered_rso[current_key] = rso[current_key]

    print(f"Number of filtered objects (perigee/apogee filter): {len(rso_keys)-len(list(filtered_rso.keys()))}")

    return filtered_rso

def compute_ax_ay(e_w_delta_arr:np.ndarray[(2,3)]) -> tuple[float]:
    ep, wp, delta_p = e_w_delta_arr[0,:]
    es, ws, delta_s = e_w_delta_arr[1,:]

    axp = ep*np.cos(wp - delta_p)
    ayp = ep*np.sin(wp - delta_p)

    axs = es*np.cos(ws - delta_s)
    ays = es*np.sin(ws - delta_s)

    return axp, ayp, axs, ays

def compute_uR(f:float, w:float, delta:float) -> float:
    return f + w - delta

def compute_min_rrel(fp0:float, fs0:float, rp:float, rs:float, iR:float, e_w_delta_arr: np.ndarray[(2,3)], maxiter:int=100, tol:float=(10*np.pi)/(3600*180)) -> tuple[float]:
    ep, wp, delta_p = e_w_delta_arr[0,:]
    es, ws, delta_s = e_w_delta_arr[1,:]
    
    # axp = ep*np.cos(wp - delta_p)
    # ayp = ep*np.sin(wp - delta_p)

    # axs = es*np.cos(ws - delta_s)
    # ays = es*np.sin(ws - delta_s)

    axp, ayp, axs, ays = compute_ax_ay(e_w_delta_arr)

    cos_iR = np.cos(iR)

    fp = fp0
    fp_prev = -np.infty

    fs = fs0
    fs_prev = -np.infty

    for i in range(maxiter):
        uR_p = compute_uR(fp, wp, delta_p)
        uR_s = compute_uR(fs, ws, delta_s)
        
        sin_uR_p, cos_uR_p = np.sin(uR_p), np.cos(uR_p)
        sin_uR_s, cos_uR_s = np.sin(uR_s), np.cos(uR_s)

        Ep = true_to_eccentric_anomaly(fp, ep)
        Es = true_to_eccentric_anomaly(fs, es)

        A = sin_uR_p + ayp
        B = cos_uR_p + axp
        C = sin_uR_s + ays
        D = cos_uR_s + axs

        coeff = A*C + B*D * cos_iR
        cos_gamma = cos_uR_p*cos_uR_s + sin_uR_p*sin_uR_s*cos_iR

        Ffp = rp*ep*np.cos(Ep) + rs*cos_gamma
        Ffs = - (rs/(1+es*np.cos(fs)))*coeff
        Gfp = - (rp/(1+ep*np.cos(fp)))*coeff
        Gfs = rs*es*np.cos(Es) + rp*cos_gamma

        F = rp*ep*np.sin(fp)+rs*(A*cos_uR_s-B*cos_iR*sin_uR_s)
        G = rs*es*np.sin(fs)+rp*(C*cos_uR_p-D*cos_iR*sin_uR_p)

        denom = Ffs*Gfp - Ffp*Gfs
        
        num_h = F*Gfs - G*Ffs
        num_k = G*Ffp - F*Gfp

        h = num_h / denom
        k = num_k / denom
        
        fp += h
        fs += k
        
        if abs(fp - fp_prev) < tol and abs(fs - fs_prev) < tol:
            break

        fp_prev = fp
        fs_prev = fs

    uR_p_final = compute_uR(fp, wp, delta_p)
    uR_s_final = compute_uR(fs, ws, delta_s)

    cos_gamma_final = np.cos(uR_p_final)*np.cos(uR_s_final)+np.sin(uR_p_final)*np.sin(uR_s_final)*np.cos(iR)

    r_rel = np.sqrt(rp**2 + rs**2 - 2*rp*rs*cos_gamma_final)

    return r_rel, fp, fs

def perigee_apogee_filter_hoots(asset:dict, rso:dict, D:float, mu:float=mu) -> dict:
    filtered_rso = {}

    state_asset = asset["state"]
    radial_p = np.linalg.norm(state_asset[:3, 0])
    a_p, e_p, i_p, aop_p, raan_p, theta_p = cartesian_to_keplerian(state_asset, mu)

    wp = np.array([np.sin(raan_p)*np.sin(i_p), np.cos(raan_p)*np.sin(i_p), np.cos(i_p)])

    for key in list(rso.keys()):
        state_object = rso[key]["state"]
        radial_s = np.linalg.norm(state_object[:3, 0])
        a_s, e_s, i_s, aop_s, raan_s, theta_s = cartesian_to_keplerian(state_object, mu)

        iR, delta_p, delta_s = compute_iR_delta_p_s(wp, [raan_p, raan_s], [i_p, i_s])

        fp0 = delta_p - aop_p
        fs0 = delta_s - aop_s
        
        e_w_delta_arr = np.array([[e_p, aop_p, delta_p], [e_s, aop_s, delta_s]])
        
        rrel_1, fp1, fs1 = compute_min_rrel(fp0, fs0, radial_p, radial_s, iR, e_w_delta_arr)
        rrel_2, fp2, fs2 = compute_min_rrel(fp1+np.pi, fs1+np.pi, radial_p, radial_s, iR, e_w_delta_arr)

        if not (rrel_1 > D and rrel_2 > D):
            filtered_rso[key] = rso[key]

    print(f"Number of filtered objects (perigee/apogee filter Hoots): {len(list(rso.keys()))-len(list(filtered_rso.keys()))}\n")

    return filtered_rso

def compute_iR_delta_p_s(wp:np.ndarray[(3,)], raan_p_s:list[float], i_p_s:list[float]) -> tuple[float]:
    raan_p, raan_s = raan_p_s
    i_p, i_s = i_p_s
    
    cos_ip = np.cos(i_p)
    sin_ip = np.sin(i_p)
    cos_is = np.cos(i_s)
    sin_is = np.sin(i_s)

    ws = np.array([np.sin(raan_s)*sin_is, np.cos(raan_s)*sin_is, cos_is])

    K = np.cross(ws, wp)
    iR = np.arcsin(np.linalg.norm(K))

    coeff_delta = 1/np.sin(iR)
    raan_diff = raan_p-raan_s
    cos_raan_diff = np.cos(raan_diff)
    sin_raan_diff = np.sin(raan_diff)

    cos_delta_p = coeff_delta*(sin_ip*cos_is-sin_is*cos_ip*cos_raan_diff)
    sin_delta_p = coeff_delta*(sin_is*sin_raan_diff)
    cos_delta_s = coeff_delta*(sin_ip*cos_is*cos_raan_diff-sin_is*cos_ip)
    sin_delta_s = coeff_delta*(sin_ip*sin_raan_diff)

    delta_p = np.arctan2(sin_delta_p, cos_delta_p)
    delta_s = np.arctan2(sin_delta_s, cos_delta_s)

    return iR, delta_p, delta_s

def generate_time_windows(key:float, window:list[float]|np.ndarray[(2,)], T:float, screening_period:float) -> np.ndarray:
    time_windows = [window]

    while time_windows[-1][0] < screening_period:
        last0, lastN = time_windows[-1]

        if last0 + T > screening_period:
           break

        if last0 > lastN:
           modified_window = [0, lastN]
           
           print(f"\nModified window for object {key}: {time_windows[-1]} -> {modified_window}")
           print(f"Window used for generating multiples: {[last0-T, lastN]}\n")

           time_windows.append([last0, lastN + T])
           time_windows[-2] = modified_window

           continue

        new_window = [last0 + T, lastN + T]
        if new_window[-1] > screening_period:
            new_window[-1] = screening_period

        time_windows.append(new_window)

    return np.array(time_windows)

def compute_T(a:float, mu:float) -> float:
    return 2 * np.pi * np.sqrt(a**3 / mu)

def compute_rhs(a:float, e:float, iR:float, ax:float, ay:float, D:float) -> tuple[float]:
    alpha = a*(1-e**2)*np.sin(iR)
    Q = alpha*(alpha - 2*D*ay) - (1-e**2)*D**2
    
    if Q < 0:
        return Q, 0, 0

    denom = alpha*(alpha - 2*D*ay) + D**2 * e**2    
    change = (alpha - D*ay)*np.sqrt(Q)

    rhs1 = (-D**2*ax + change) / denom
    rhs2 = (-D**2*ax - change) / denom

    return Q, rhs1, rhs2

def generate_dM(window:list[float], M0:float) -> np.ndarray:
    dM = np.array([M_val+ np.pi - M0 for M_val in window])
    
    if np.any(dM < 0):
       dM += 2*np.pi
       dM %= (2*np.pi)

    return dM

def compute_uR_1_2(rhs:float) -> tuple[float]:
    uR1 = np.arccos(rhs)

    return uR1, 2*np.pi - uR1

def time_filter(asset:dict, rso:dict, D:float, period:float, mu:float=mu) -> dict:
    filtered_rso = {}

    a_p, e_p, i_p, aop_p, raan_p, theta_p = cartesian_to_keplerian(asset["state"], mu)
    T_p = compute_T(a_p, mu)
    M0_p = true_to_mean_anomaly(e_p, theta_p)
    if M0_p < 0:
       M0_p += np.pi

    wp = np.array([np.sin(raan_p)*np.sin(i_p), np.cos(raan_p)*np.sin(i_p), np.cos(i_p)])

    for key in list(rso.keys()):
        a_s, e_s, i_s, aop_s, raan_s, theta_s = cartesian_to_keplerian(rso[key]["state"], mu)
        T_s = compute_T(a_s, mu)
        
        M0_s = true_to_mean_anomaly(e_s, theta_s)
        if M0_s < 0:
           M0_s += np.pi

        iR, delta_p, delta_s = compute_iR_delta_p_s(wp, [raan_p, raan_s], [i_p, i_s])

        axp, ayp, axs, ays = compute_ax_ay(np.array([[e_p, aop_p, delta_p], [e_s, aop_s, delta_s]]))

        Qp, rhs_p1, rhs_p2 = compute_rhs(a_p, e_p, iR, axp, ayp, D)
        Qs, rhs_s1, rhs_s2 = compute_rhs(a_s, e_s, iR, axs, ays, D)

        if Qp < 0 or Qs < 0 or np.any(np.abs(np.array([rhs_p1, rhs_p2, rhs_s1, rhs_s2]))>1):
            filtered_rso[key] = rso[key]
            continue

        uR_p11, uR_p12 = compute_uR_1_2(rhs_p1)
        uR_p21, uR_p22 = compute_uR_1_2(rhs_p2)
        uR_s11, uR_s12 = compute_uR_1_2(rhs_s1)
        uR_s21, uR_s22 = compute_uR_1_2(rhs_s2)

        increment_p = - aop_p + delta_p
        increment_s = - aop_s + delta_s
        
        window_p1 = [true_to_mean_anomaly(e_p, uR_p12+increment_p), true_to_mean_anomaly(e_p, uR_p11+increment_p)]
        window_p2 = [true_to_mean_anomaly(e_p, uR_p21+increment_p), true_to_mean_anomaly(e_p, uR_p22+increment_p)]
        window_s1 = [true_to_mean_anomaly(e_s, uR_s12+increment_s), true_to_mean_anomaly(e_s, uR_s11+increment_s)]
        window_s2 = [true_to_mean_anomaly(e_s, uR_s21+increment_s), true_to_mean_anomaly(e_s, uR_s22+increment_s)]

        dM_p1, dM_p2 = generate_dM(window_p1, M0_p), generate_dM(window_p2, M0_p)
        dM_s1, dM_s2 = generate_dM(window_s1, M0_s), generate_dM(window_s2, M0_s)

        dM_p = np.sort(np.hstack((dM_p1, dM_p2)))
        dM_s = np.sort(np.hstack((dM_s1, dM_s2)))

        if dM_p[1]-dM_p[0]>1.5 and dM_p[-1]-dM_p[-2]>1.5 and dM_p[2]-dM_p[1] < 1.5:
           dM_p = np.array([dM_p[-1]]+list(dM_p[:-1]))
        
        if dM_s[1]-dM_s[0]>1.5 and dM_s[-1]-dM_s[-2]>1.5 and dM_s[2]-dM_s[1] < 1.5:
           dM_s = np.array([dM_s[-1]]+list(dM_s[:-1]))
        
        window_p1 = [delta_mean_anomaly_to_elapsed_time(dM_val_p1, mu, a_p) for dM_val_p1 in dM_p[:2]]
        window_p2 = [delta_mean_anomaly_to_elapsed_time(dM_val_p2, mu, a_p) for dM_val_p2 in dM_p[2:]]
        window_s1 = [delta_mean_anomaly_to_elapsed_time(dM_val_s1, mu, a_s) for dM_val_s1 in dM_s[:2]]
        window_s2 = [delta_mean_anomaly_to_elapsed_time(dM_val_s2, mu, a_s) for dM_val_s2 in dM_s[2:]]

        time_windows_p1 = generate_time_windows(key, window_p1, T_p, period)
        time_windows_p2 = generate_time_windows(key, window_p2, T_p, period)
        windows_p = np.vstack((time_windows_p1, time_windows_p2))
       
        time_windows_s1 = generate_time_windows(key, window_s1, T_s, period)
        time_windows_s2 = generate_time_windows(key, window_s2, T_s, period)
        windows_s = np.vstack((time_windows_s1, time_windows_s2))

        for k in range(windows_p.shape[0]):
            cw_p = windows_p[k, :]
            overlap_found = False
            
            for l in range(windows_s.shape[0]):
                cw_s = windows_s[l, :]

                if np.max([cw_p[0], cw_s[0]]) - 3.5 <= np.min([cw_p[1], cw_s[1]]) + 3.5:
                    filtered_rso[key] = rso[key]
                    overlap_found = True
                    break
            
            if overlap_found:
                break

    print(f"Number of filtered objects (time filter Hoots): {len(list(rso.keys()))-len(list(filtered_rso.keys()))}\n")

    return filtered_rso

def print_encapsulated(text:str, char_top_bottom:chr="-", char_sides:chr="|") -> None:
    str_top_bottom = (len(text)+4) * char_top_bottom
    
    print(str_top_bottom) 
    print(f"{char_sides} {text} {char_sides}")
    print(str_top_bottom) 

    return None

if __name__ == "__main__":

    #################### INPUTS ####################
    group_number     = 1
    D                = 100E3    # [m]
    screening_period = 48*3600  # [s]
    ################################################

    group2asset_NORAD_ID = {
        1: 36508,
        2: 39159,
        3: 39452,
        4: 31698,
        5: 40697,
        6: 45551,
    }
    asset_NORAD_ID = group2asset_NORAD_ID[group_number]

    rso_dict = cu.read_catalog_file(f"data/group{group_number}/estimated_rso_catalog.pkl")

    asset = rso_dict[asset_NORAD_ID]
    rso_dict.pop(asset_NORAD_ID)

    filtered = perigee_apogee_filter(asset, rso_dict, D)
    print(f"Remaining objects after the perigee/apogee filter:\n{sorted(filtered.keys())}\n")

    filtered_rso = perigee_apogee_filter_hoots(asset, filtered, D)

    print(f"Remaining objects after the perigee/apogee filter from Hoots:\n{sorted(filtered_rso.keys())}\n")

    filtered_rso_final = time_filter(asset, filtered_rso, D, screening_period)

    print_encapsulated(f"Remaining objects: {list(filtered_rso_final.keys())}")

    with open("filtered_catalog.pkl", "wb") as f:
        filtered_rso_final[asset_NORAD_ID] = asset

        pickle.dump(filtered_rso_final, f)

    result_simone = sorted([91000, 91452, 91216, 91270, 91132, 91410, 91425, 91054, 91047, 91444, 91068, 91049])

    overlapping = []
    different = []
    for i, val in enumerate(filtered_rso_final):
        if val in result_simone:
            overlapping.append(val)
            result_simone.remove(val)
            continue

        different.append(val)     

    print(f"\nOverlapping objects: {overlapping}")
    print(f"Extra objects in Bora: {different}")
    print(f"Extra objects in Simone: {result_simone}")
