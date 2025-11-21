# import numpy as np

# ref_path = "short_reference_gait.npz"

# ref = np.load(ref_path, allow_pickle=True)
# print("keys in npz:", list(ref.keys()))

# metadata = ref["metadata"].item()   # 0-d object array → Python object
# series_data = ref["series_data"].item()

# print("\n[metadata] type:", type(metadata))
# if isinstance(metadata, dict):
#     print("metadata keys:", metadata.keys())

# print("\n[series_data] type:", type(series_data))
# if isinstance(series_data, dict):
#     print("series_data keys:", series_data.keys())

# for k, v in series_data.items():
#     if hasattr(v, "shape"):
#         print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
#     else:
#         print(f"  {k}: type={type(v)}")



import h5py
import numpy as np

h5_path   = "S004_level_08mps.h5" 
ref_npz   = "short_reference_gait.npz"
out_npz   = "S004_level_08mps.npz"

ref = np.load(ref_npz, allow_pickle=True)

ref_metadata = ref["metadata"].item()
ref_series   = ref["series_data"].item()

print("reference metadata type :", type(ref_metadata))
print("reference series_data type :", type(ref_series))

if not isinstance(ref_metadata, dict):
    raise TypeError("metadata는 dict가 아닌 다른 타입입니다. 코드에 맞게 수정 필요.")
if not isinstance(ref_series, dict):
    raise TypeError("series_data는 dict가 아닌 다른 타입입니다. 코드에 맞게 수정 필요.")

with h5py.File(h5_path, "r") as h5:

    new_metadata = dict(ref_metadata) 
    
    if "subject_id" in new_metadata:
        new_metadata["subject_id"] = "S004"
    if "trial_name" in new_metadata:
        new_metadata["trial_name"] = "level_08mps"
    if "sample_rate" in new_metadata:
        new_metadata["sample_rate"] = 100.0
    if "data_length" in new_metadata:
        new_metadata["data_length"] = len(h5['MoCap']['kin_q']['time'])

    new_series = {}
    mapping = {
        "time"        : "MoCap/kin_q/time",
        "q_pelvis_tx"  : "MoCap/kin_q/pelvis_tx",
        "q_pelvis_tz"  : "MoCap/kin_q/pelvis_tz",
        "q_pelvis_ty"  : "MoCap/kin_q/pelvis_ty",
        "q_pelvis_tilt"  : "MoCap/kin_q/pelvis_tilt",
        "q_pelvis_list"  : "MoCap/kin_q/pelvis_list",
        "q_pelvis_rotation"  : "MoCap/kin_q/pelvis_rotation",
        "q_hip_flexion_r"  : "MoCap/kin_q/hip_flexion_r",
        "q_hip_adduction_r"  : "MoCap/kin_q/hip_adduction_r",
        "q_hip_rotation_r"  : "MoCap/kin_q/hip_rotation_r",
        "q_knee_angle_r"  : "MoCap/kin_q/knee_angle_r",
        "q_ankle_angle_r"  : "MoCap/kin_q/ankle_angle_r",
        "q_hip_flexion_l":  "MoCap/kin_q/hip_flexion_l",
        "q_hip_adduction_l":  "MoCap/kin_q/hip_adduction_l",
        "q_hip_rotation_l":  "MoCap/kin_q/hip_rotation_l",
        "q_knee_angle_l":  "MoCap/kin_q/knee_angle_l",
        "q_ankle_angle_l":  "MoCap/kin_q/ankle_angle_l",

        "dq_pelvis_tx":  "Common/v_Y_true",
        "dq_pelvis_tz":  "MoCap/kin_qdot/pelvis_tz",
        "dq_pelvis_ty":  "MoCap/kin_qdot/pelvis_ty",
        "dq_pelvis_tilt":  "MoCap/kin_qdot/pelvis_tilt",
        "dq_pelvis_list":  "MoCap/kin_qdot/pelvis_list",
        "dq_pelvis_rotation":  "MoCap/kin_qdot/pelvis_rotation",
        "dq_hip_flexion_r":  "MoCap/kin_qdot/hip_flexion_r",
        "dq_hip_adduction_r":  "MoCap/kin_qdot/hip_adduction_r",
        "dq_hip_rotation_r":  "MoCap/kin_qdot/hip_rotation_r",
        "dq_knee_angle_r":  "MoCap/kin_qdot/knee_angle_r",
        "dq_ankle_angle_r":  "MoCap/kin_qdot/ankle_angle_r",
        "dq_hip_flexion_l":  "MoCap/kin_qdot/hip_flexion_l",
        "dq_hip_adduction_l":  "MoCap/kin_qdot/hip_adduction_l",
        "dq_hip_rotation_l":  "MoCap/kin_qdot/hip_rotation_l",
        "dq_knee_angle_l":  "MoCap/kin_qdot/knee_angle_l",
        "dq_ankle_angle_l":  "MoCap/kin_qdot/ankle_angle_l"
    }
    
    ANGLE_Q_KEYS_DEG_TO_RAD = [
        "q_pelvis_tilt",
        "q_pelvis_list",
        "q_pelvis_rotation",
        "q_hip_flexion_r","q_hip_adduction_r","q_hip_rotation_r",
        "q_knee_angle_r","q_ankle_angle_r",
        "q_hip_flexion_l","q_hip_adduction_l","q_hip_rotation_l",
        "q_knee_angle_l","q_ankle_angle_l"
    ]

    ANGLE_DQ_KEYS_DEG_TO_RAD = [
        "dq_pelvis_tilt",
        "dq_pelvis_list",
        "dq_pelvis_rotation",
        "dq_hip_flexion_r","dq_hip_adduction_r","dq_hip_rotation_r",
        "dq_knee_angle_r","dq_ankle_angle_r",
        "dq_hip_flexion_l","dq_hip_adduction_l","dq_hip_rotation_l",
        "dq_knee_angle_l","dq_ankle_angle_l"
    ]

    for key in mapping.keys():
        h5_path_for_key = mapping[key]
        if h5_path_for_key not in h5:
            print(f"경고: H5 파일에 '{h5_path_for_key}' 경로가 없습니다. '{key}' 키를 건너뜁니다.")
            continue

        data = np.array(h5[h5_path_for_key])  # h5 dataset → numpy array
        if key in ANGLE_Q_KEYS_DEG_TO_RAD or key in ANGLE_DQ_KEYS_DEG_TO_RAD:
            if key == "q_hip_adduction_l" or key == "dq_hip_adduction_l":
                print(f"Converting {key} from degrees to radians with sign inversion.")
                data = -np.deg2rad(data) 
            else:
                data = np.deg2rad(data)

        new_series[key] = data
        print(f"filled series_data['{key}'] from h5['{h5_path_for_key}'], shape={data.shape}")

metadata_arr    = np.zeros((), dtype=object)
series_data_arr = np.zeros((), dtype=object)

metadata_arr[()]    = new_metadata
series_data_arr[()] = new_series

np.savez(out_npz, metadata=metadata_arr, series_data=series_data_arr)

print(f"\nSaved converted npz to: {out_npz}")
