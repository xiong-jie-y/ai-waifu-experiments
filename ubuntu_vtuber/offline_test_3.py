import open3d as o3d

import numpy as np
import pandas as pd
import json

def get_dfs():
    face_landmark_frames = json.load(open("face_landmark_frames_20200425-220448.json", "r"))
    # import IPython; IPython.embed()
    return [np.array(face_landmark_frames[0][0]['landmarks']) for face_landmark_frame in face_landmark_frames]

dfs = get_dfs()
df = dfs[0]
df = pd.DataFrame(df, columns=['x','y','z'])

import plotly.express as px
fig = px.histogram(df, x="z")
# fig.show()

import streamlit as st
st.write(fig)