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

file_path = st_files.choose_file_from_glob("logs", "hand**/Hand*.json")

first_landmarks = flu.get_landmark_from_face_frames_file(file_path)
index = st.slider('Frame Number', 0, len(first_landmarks), 1)
first_landmark = first_landmarks[index]

st_human_state.draw_landmark_3d_with_index_v2(first_landmark)
st_human_state.draw_landmark_2d_with_index(first_landmark)