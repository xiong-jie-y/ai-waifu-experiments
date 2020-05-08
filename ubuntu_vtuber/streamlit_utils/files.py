import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import open3d as o3d
import streamlit as st


def choose_file_from_glob(root_path, init_glob):
    glob_str = st.sidebar.text_input("Glob", init_glob)
    files = list(glob.glob(os.path.join(root_path, glob_str), recursive=True))
    return st.sidebar.selectbox('choose your log file', files)
