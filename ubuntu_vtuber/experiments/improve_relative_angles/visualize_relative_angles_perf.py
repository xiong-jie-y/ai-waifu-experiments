import streamlit_utils.development as st_dev

args = [
    (landmark, flu.FINGER_IDS["index_finger"]) 
    for landmark in first_landmarks]

st_dev.compare_function_performances(
    [
        flu.get_relative_angles_from_wrist_v1,
        flu.get_relative_angles_from_wrist_v2,
        flu.get_relative_angles_from_wrist_v3,
        flu.get_relative_angles_from_wrist_v4], 
    args)

st_dev.inspect_function(flu.get_relative_angles_from_wrist_v1, (first_landmark, flu.FINGER_IDS["index_finger"]))
st_dev.inspect_function(flu.get_relative_angles_from_wrist_v2, (first_landmark, flu.FINGER_IDS["index_finger"]))
