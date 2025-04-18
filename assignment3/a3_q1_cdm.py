import pickle

def CDM(data:dict, headers:list[str], unit_distance:str="m", unit_velocity:str="m/s", space_header:int=10, space_unit:int=2) -> str:
    header_max_length = max([len(header) for header in headers])
    value_max_length = max([len(str(val)) for val in list(data.values())])

    header_len = header_max_length + space_header

    key2unit = {
        "EUCLIDEAN_MISS_DISTANCE": unit_distance, 
        "RELATIVE_SPEED": unit_velocity,
        "RELATIVE_POSITION_R": unit_distance, 
        "RELATIVE_POSITION_T": unit_distance, 
        "RELATIVE_POSITION_N": unit_distance, 
        "RELATIVE_VELOCITY_R": unit_velocity, 
        "RELATIVE_VELOCITY_T": unit_velocity, 
        "RELATIVE_VELOCITY_N": unit_velocity
    }
    keys_with_unit = list(key2unit.keys())

    sorted_data = {key: data[key] for key in headers}

    cdm = ""

    for key, value in sorted_data.items():
        add_unit_str = ''

        if str(value)[1:] == "0.0":
            value = 0.0
       
        if key in keys_with_unit:
            add_unit_str = (value_max_length - len(str(value)))*' ' + space_unit*' ' + f'[{key2unit[key]}]'

        cdm += f"{key}{(header_len - len(key))*' '}: {value}{add_unit_str}\n"

    return cdm

if __name__ == "__main__":

    headers = ["CREATION_DATE", "ORIGINATOR", "MESSAGE_FOR", "MESSAGE_ID", "TCA", "EUCLIDEAN_MISS_DISTANCE", "RELATIVE_SPEED", "COLLISION_PROBABILITY", "UPPER_COLLISION_PROBABILITY", "MAHALANOBIS_MISS_DISTANCE", "RELATIVE_POSITION_R", "RELATIVE_POSITION_T", "RELATIVE_POSITION_N", "RELATIVE_VELOCITY_R", "RELATIVE_VELOCITY_T", "RELATIVE_VELOCITY_N", "START_SCREEN_PERIOD", "END_SCREEN_PERIOD"]

    pklFile = open("cdm_data.pkl", 'rb' )
    cdm_data = pickle.load( pklFile )
    pklFile.close()

    asset_NORAD_ID = cdm_data["asset_NORAD_ID"]
    constant_entries = cdm_data["constant_entries"]
    cdm_data.pop("asset_NORAD_ID")
    cdm_data.pop("constant_entries")

    objects = list(cdm_data.keys())
    print(f"Objects: {objects}\n")

    print(f"CDM data for: {asset_NORAD_ID}")
    print(f"Constant entries: {constant_entries}\n")

    cdms = {}

    for key in objects:
        conjuntions_per_object = cdm_data[key]
        ids = list(conjuntions_per_object.keys())

        for id in ids:
            conjuntion_params = conjuntions_per_object[id]

            conjuntion_cdm = CDM(constant_entries | {"MESSAGE_ID": id} | conjuntion_params, headers)
            
            cdms[id] = conjuntion_cdm
            print(conjuntion_cdm)

    with open("cdms.pkl", "wb") as f:
        pickle.dump(cdms, f)
