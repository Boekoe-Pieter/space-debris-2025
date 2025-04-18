import time
import pickle
import numpy as np
import TudatPropagator as tp
import ConjunctionUtilities as cu
from datetime import datetime, timedelta
import tudatpy.astro.time_conversion as tc
from tudatpy.interface import spice

def print_encapsulated(text:str, char_top_bottom:chr="-", char_sides:chr="|") -> None:
    str_top_bottom = (len(text)+4) * char_top_bottom
    
    print(str_top_bottom) 
    print(f"{char_sides} {text} {char_sides}")
    print(str_top_bottom) 

    return None

def state_params(object:dict) -> dict:
    return {
        'central_bodies': ['Earth'],
        'bodies_to_create': ['Earth', 'Moon', 'Sun'],
        'mass': object['mass'],
        'Cd': object['Cd'],
        'Cr': object['Cr'],
        'area': object['area'],
        'sph_deg': 8,
        'sph_ord': 8
    }

def compute_radius(A:float) -> float:
    return np.sqrt( A / (4 * np.pi) )

def rM(rA:np.ndarray[(3,1)], PA:np.ndarray[(3,3)], rB:np.ndarray[(3,1)], PB:np.ndarray[(3,3)]) -> float:
    rdiff = rA - rB

    return np.sqrt(rdiff.T @ np.linalg.inv(PA + PB) @ rdiff)[0,0]

def TDB2UTC(converter:tc.TimeScaleConverter, time_TDB:float) -> float:
    earth_pos = spice.get_body_cartesian_state_at_epoch(
        target_body_name = "Earth",
        observer_body_name = "Sun",
        reference_frame_name = "ECLIPJ2000",
        aberration_corrections = "NONE",
        ephemeris_time = time_TDB,
    )

    return converter.convert_time(
        input_scale  = tc.TimeScales.tdb_scale, 
        output_scale = tc.TimeScales.utc_scale, 
        input_value  = time_TDB, 
        earth_fixed_position = earth_pos[:3][:,np.newaxis]
    )

def elapsed_seconds2datetime(ref_time:datetime, elapsed_seconds:float) -> datetime:
    return ref_time + timedelta(seconds=elapsed_seconds)

if __name__ == "__main__":

    #################### INPUTS ####################
    group_number     = 1
    D                = 100E3    # [m]
    screening_period = 48*3600  # [s]
    miss_distance    = 1E3      # [m]

    start_time = datetime(2025, 4, 1, 12, 0, 0)
    int_params = {
        'tudat_integrator': 'rkf78',
        'step': 10.0,
        'max_step': 1000.0,
        'min_step': 1e-3,
        'rtol': 1e-10,
        'atol': 1e-10,
    }
    ################################################

    ref_time = datetime(2000, 1, 1, 12, 0, 0)
    initial_epoch = (start_time - ref_time).total_seconds()
    end_time = start_time+timedelta(seconds=screening_period)
    trange = [initial_epoch, (end_time - ref_time).total_seconds()]
    
    # converter = tc.default_time_scale_converter()

    # start_screening_datetime = elapsed_seconds2datetime(ref_time, TDB2UTC(converter, trange[0]))
    # final_screening_datetime = start_screening_datetime + timedelta(seconds=screening_period)

    group2asset_NORAD_ID = {
        1: 36508,
        2: 39159,
        3: 39452,
        4: 31698,
        5: 40697,
        6: 45551,
    }
    asset_NORAD_ID = group2asset_NORAD_ID[group_number]

    pklFile = open("filtered_catalog.pkl", 'rb' )
    rso_dict = pickle.load( pklFile )
    pklFile.close()    

    asset = rso_dict[asset_NORAD_ID]
    asset_state = asset["state"]
    asset_covar = asset["covar"]
    asset_r = compute_radius(asset["area"])
    rso_dict.pop(asset_NORAD_ID)

    asset_params = state_params(asset)

    object_keys = list(rso_dict.keys())
    conjuntions = {}
    no_total_conjunctions = 0

    for key in object_keys:
        start_calc = time.time()

        print(f"Object {key}")
        object = rso_dict[key]

        object_params = state_params(object)

        T_list, rho_list = cu.compute_TCA(asset_state, object["state"], trange, asset_params, object_params, int_params, rho_min_crit=miss_distance)
        no_conjuntion = len(rho_list)

        if no_conjuntion == 1 and rho_list[0] > miss_distance:
            print(f"Miss distance even at TCA ({round(rho_list[0], 3)} [m]) is higher than the threshold value of {miss_distance} [m]")
            print(f"Calculation time: {round(time.time()-start_calc, 3)} [s]\n")
            continue

        for i in range(no_conjuntion):
            print(f"Hours elapsed since initial epoch: {round((T_list[i]-initial_epoch)/3600, 3)} [hr]")
            print(f'Miss distance: {round(rho_list[i], 3)} [m]')

        print(f"Number of viable conjuctions: {no_conjuntion}")
        conjuntions[key] = np.array([T_list, rho_list])
        no_total_conjunctions += no_conjuntion

        print(f"Calculation time: {round(time.time()-start_calc, 3)} [s]\n")

    print_encapsulated(f"Total number of conjuctions: {no_total_conjunctions}")
    print(f"Viable conjunctions: {conjuntions}\n")

    cdm = {}

    for key in list(conjuntions.keys()):
        object = rso_dict[key]
        object_params = state_params(object)
        object_r = compute_radius(object["area"])
        
        HBR_current = asset_r + object_r
        
        T_rho = conjuntions[key]
        T_arr, rho_arr = T_rho[0, :], T_rho[1, :]

        cdm_id = {}
        for index, T_val in enumerate(T_arr):
            trange_current = [trange[0], T_val]

            message_ID = f"{asset_NORAD_ID}_{key}_{index+1}"
            tca_object = elapsed_seconds2datetime(ref_time, T_val)
            
            tf_a, state_a, covar_a = tp.propagate_state_and_covar(asset_state, asset_covar, trange_current, asset_params, int_params)
            
            tf_o, state_o, covar_o = tp.propagate_state_and_covar(object["state"], object["covar"], trange_current, object_params, int_params)

            state_rel = state_o - state_a
            rrel = state_rel[:3, :]
            vrel = state_rel[3:, :]

            rrel_rtn = cu.eci2ric(rrel, vrel, rrel)[:,0]
            vrel_rtn = cu.eci2ric_vel(rrel, vrel, rrel_rtn, vrel)[:,0]
            
            eu_miss_object = np.linalg.norm(rrel_rtn)
            rel_speed = np.linalg.norm(vrel_rtn)

            # print(np.linalg.norm(state_a[:3,0]-state_o[:3,0]), rho_arr[index])
            
            pc_object = cu.Pc2D_Foster(state_a, covar_a, state_o, covar_o, HBR_current)
            uc_object = cu.Uc2D(state_a, covar_a, state_o, covar_o, HBR_current)
            rM_object = rM(state_a[:3,:], covar_a[:3,:3], state_o[:3,:], covar_o[:3,:3])

            cdm_conjunction = {}
            cdm_conjunction["CREATION_DATE"] = datetime.now()
            cdm_conjunction["TCA"] = tca_object

            cdm_conjunction["EUCLIDEAN_MISS_DISTANCE"] = round(eu_miss_object, 6)
            cdm_conjunction["RELATIVE_SPEED"] = round(rel_speed, 9)
            cdm_conjunction["COLLISION_PROBABILITY"] = pc_object
            cdm_conjunction["UPPER_COLLISION_PROBABILITY"] = uc_object
            cdm_conjunction["MAHALANOBIS_MISS_DISTANCE"] = round(rM_object, 6)
            
            cdm_conjunction["RELATIVE_POSITION_R"] = round(rrel_rtn[0], 6)
            cdm_conjunction["RELATIVE_POSITION_T"] = round(rrel_rtn[1], 6)
            cdm_conjunction["RELATIVE_POSITION_N"] = round(rrel_rtn[2], 6)
            cdm_conjunction["RELATIVE_VELOCITY_R"] = round(vrel_rtn[0], 6)
            cdm_conjunction["RELATIVE_VELOCITY_T"] = round(vrel_rtn[1], 6)
            cdm_conjunction["RELATIVE_VELOCITY_N"] = round(vrel_rtn[2], 6)

            cdm_id[message_ID] = cdm_conjunction

        cdm[key] = cdm_id

    cdm["asset_NORAD_ID"] = asset_NORAD_ID
    cdm["constant_entries"] = {"ORIGINATOR": f"Group {group_number}",   
                               "MESSAGE_FOR": "Cryosat2",
                               "START_SCREEN_PERIOD": start_time,
                               "END_SCREEN_PERIOD": end_time
                            }

    with open("cdm_data.pkl", "wb") as f:
        pickle.dump(cdm, f)
