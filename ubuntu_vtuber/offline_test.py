import streamlit as st

"""
PointCloudLayer
===============
A subset of a Creative Commons-licensed laser-scanned point cloud of Choshi-Otaki Falls in Aomori, Japan.

The full data set is viewable here:
https://sketchfab.com/3d-models/choshi-otaki-falls-oirase-valley-aomori-ea1ef9e7f82f418ea0776ceb6894ebd1
"""

import pydeck as pdk
import pandas as pd

import numpy as np

import pydeck
import pandas as pd

import json

@st.cache
def get_dfs():
    face_landmark_frames = json.load(open("face_landmark_frames_20200425-220448.json", "r"))
    # import IPython; IPython.embed()
    return [pd.DataFrame(face_landmark_frames[0][0]['landmarks'], columns=['x', 'y', 'z']) for face_landmark_frame in face_landmark_frames]

dfs = get_dfs()

i = 0
df = dfs[i]
target = [df.x.mean(), df.y.mean(), df.z.mean()]

print(df)
# import IPython; IPython.embed()

point_cloud_layer = pydeck.Layer(
    "PointCloudLayer",
    data=df,
    get_position=["x", "y", "z"],
    auto_highlight=True,
    pickable=True,
    point_size=3,
)

view_state = pydeck.ViewState(target=target, controller=True, rotation_x=15, rotation_orbit=30, zoom=5.3)
view = pydeck.View(type="OrbitView", controller=True)

r = pydeck.Deck(point_cloud_layer, initial_view_state=view_state, views=[view])

st.pydeck_chart(r)