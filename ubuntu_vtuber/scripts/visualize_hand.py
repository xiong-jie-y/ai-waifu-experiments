import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import streamlit as st
from mpl_toolkits.mplot3d import Axes3D

import facelandmark_utils as flu
import streamlit_utils.files as st_files
import streamlit_utils.human_state as st_human_state

import zmq

@st.cache(hash_funcs={zmq.sugar.socket.Socket: lambda _: None})
def get_socket():
    context = zmq.Context()
    publisher = context.socket(zmq.PUB)
    publisher.bind("tcp://127.0.0.1:9998")
    return publisher

def select_from_functions(funcs):
    func_map = {}
    for func in funcs:
        func_map[func.__name__] = func

    mode = st.sidebar.radio("what to explore", list(func_map.keys()))
    func_map[mode]()

def see_multiple_landmarks():
    file_paths = st_files.choose_multi_file_from_glob("logs", "hand**/Hand*.json")

    chosen_landmarks = []
    first = True
    for file_path in file_paths:
        first_landmarks = flu.get_landmark_from_face_frames_file(file_path)
        if first:
            index = st.slider('Frame Number', 0, len(first_landmarks), 1)
            first = False
        chosen_landmarks.append(first_landmarks[index])

    st_human_state.draw_landmark_3d_with_index_v3(chosen_landmarks)

import streamlit_utils.development as st_dev
from scipy.spatial.transform import Rotation

import msgpack

def see_landmark():
    file_path = st_files.choose_file_from_glob("logs", "hand**/Hand*.json")

    # first_landmarks = flu.get_landmark_from_face_frames_file(file_path)
    # index = st.slider('Frame Number', 0, len(first_landmarks), 1)
    # st.write("sending")
    publisher = get_socket()
    # tracker = flu.RightLeftHandStateTracker()
    # tracker.get_left_right_hand_states()

    # landmark = first_landmarks[index]
    landmark = flu.choose_landmark_from_file(file_path)
    normalized_graph = st.empty()

    msg = {}
    msg["HandLandmarks"] = {}
    direction = st.radio("Choose which hand", ["right", "left"])
    msg["HandLandmarks"][direction] = flu.get_hand_state(landmark)
    data = ["HumanState".encode('utf-8'), msgpack.packb(msg, use_bin_type=True)]
    publisher.send_multipart(data)

    st_dev.inspect_function(flu.get_wrist_angle, (landmark, ))
    # flu.get_wrist_angle(landmark)
    # st_dev.compare_function_performances(
    #     [flu.get_wrist_angle],
    #     [(lm,) for lm in first_landmarks]
    # )

    # st_dev.write_as_json(np.cross([1, 0, 0], [0, 1, 0]))
    rot = Rotation.from_quat(flu.get_wrist_angle(landmark))
    # st_human_state.get_landmark_3d_with_index_v2(landmark)
    st_human_state.draw_landmark_3d_with_index_v2(landmark)
    # normalized_landmark = rot.inv().apply(landmark)
    normalized_landmark = rot.inv().apply(landmark)
    st_dev.inspect_function(flu.get_relative_angles_from_xy_plain, (normalized_landmark[flu.FINGER_IDS['index_finger']], ))

    st_dev.inspect_function(
        flu.get_shortest_rotvec_between_two_vector,
        (np.array([2,0,0]), np.array(([0, 1,0])))
    )

    st_dev.inspect_function(
        flu.get_relative_angles_from_xy_plain,
        (np.array([[0,0,0], [1,1,1.0], [2,2,2]]),))

    fig = st_human_state.get_landmark_3d_with_index_v2(normalized_landmark)
    normalized_graph.plotly_chart(fig)
    st_dev.inspect_function(flu.get_hand_state, (landmark, ))

def see_z_distance():
    file_path = st_files.choose_file_from_glob("logs", "hand**/Hand*.json")
    first_landmarks = flu.get_landmark_from_face_frames_file(file_path)
    stds = []

    for i in range(0, len(first_landmarks[0])):
        stds.append((i, np.std([landmark[i][2] for landmark in first_landmarks])))
        st_human_state.draw_wrist_z_change(first_landmarks, i)

    stds.sort(key=lambda t: t[1])
    for std in stds:
        st.write(std)

    st_human_state.draw_wrist_z_change(first_landmarks, 305)

select_from_functions([see_multiple_landmarks, see_landmark, see_z_distance])
