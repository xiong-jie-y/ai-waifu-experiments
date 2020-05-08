import open3d as o3d

import numpy as np
import json

import argparse
import streamlit as st

import os
import glob

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import facelandmark_utils as flu
import streamlit_utils.files as st_files

import plotly.graph_objects as go

def draw_landmark_3d_with_index_v2(landmark):
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=landmark[:,0],
        y=landmark[:,1],
        z=landmark[:,2],
        mode='markers'
    ))
    fig.update_layout(
        scene=dict(
            aspectratio=dict(
                x=1,
                y=1,
                z=1
            ),
            annotations=[
                dict(
                    showarrow=False,
                    xanchor="left",
                    xshift=10,
                    x=x,
                    y=y,
                    z=z,
                    text=str(index)
                ) for index, (x,y,z) in enumerate(landmark)
            ]
        )
    )
    # fig.update_layout(
    #     xaxis=dict(title_text="x"),
    #     yaxis=dict(title_text="y"),
    #     scene=dict(
    #         aspectratio=dict(
    #             x=1,
    #             y=1,
    #             z=1
    #         ),
    #         camera=dict(
    #             center=dict(
    #                 x=0,
    #                 y=0,
    #                 z=0
    #             ),
    #             eye=dict(
    #                 x=1.96903462608,
    #                 y=-1.09022831971,
    #                 z=0.405345349304
    #             ),
    #             up=dict(
    #                 x=0,
    #                 y=0,
    #                 z=1
    #             )
    #         ),
    #         dragmode="turntable",
    #         annotations=[dict(
    #             showarrow=False,
    #             x="2017-01-01",
    #             y="A",
    #             z=0,
    #             text="Point 1",
    #             xanchor="left",
    #             xshift=10,
    #             opacity=0.7
    #         ), dict(
    #             x="2017-02-10",
    #             y="B",
    #             z=4,
    #             text="Point 2",
    #             textangle=0,
    #             ax=0,
    #             ay=-75,
    #             font=dict(
    #                 color="black",
    #                 size=12
    #             ),
    #             arrowcolor="black",
    #             arrowsize=3,
    #             arrowwidth=1,
    #             arrowhead=1
    #         ), dict(
    #             x="2017-03-20",
    #             y="C",
    #             z=5,
    #             ax=50,
    #             ay=0,
    #             text="Point 3",
    #             arrowhead=1,
    #             xanchor="left",
    #             yanchor="bottom"
    #         )]
    #     ),
    # )

    st.write(fig)

def draw_landmark_3d_with_index(landmark, filter_ids=[]):
    elev = st.slider('Elev', 0, 360, 90)
    azim = st.slider('Azim', 0, 360, 90)

    fig = plt.figure()
    ax1 = fig.add_subplot(111 , projection='3d')
    ax1.view_init(elev=elev, azim=azim)

    for i, keypoint in enumerate(landmark):
        if len(filter_ids) == 0 or i in filter_ids:
            ax1.text(keypoint[0], keypoint[1], keypoint[2], s=str(i))

    # direction = flu.simple_face_direction(landmark)
    # direction = direction / np.linalg.norm(direction)
    # ax1.plot(direction[:,0], direction[:,1], direction[:,2])
    if len(filter_ids) == 0:
        ax1.scatter(landmark[:,0],landmark[:,1], zs=landmark[:,2])
    else:
        ax1.scatter(landmark[filter_ids,0],landmark[filter_ids,1], zs=landmark[filter_ids,2])
    st.write(fig)

def draw_landmark_2d_with_index(landmark, filter_ids=[]):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    for i, keypoint in enumerate(landmark):
        if len(filter_ids) == 0 or i in filter_ids:
            ax1.text(keypoint[0], keypoint[1], s=str(i))

    # direction = flu.simple_face_direction(landmark)
    # direction = direction / np.linalg.norm(direction)
    # ax1.plot(direction[:,0], direction[:,1], direction[:,2])
    if len(filter_ids) == 0:
        ax1.scatter(landmark[:,0],landmark[:,1])
    else:
        ax1.scatter(landmark[filter_ids,0],landmark[filter_ids,1])
    st.write(fig)
    # plt.show()

def draw_landmark(landmark):
    points = o3d.utility.Vector3dVector(first_landmark)
    point_cloud = o3d.geometry.PointCloud(points)
    colors = [[float(i) / len(points), 0, 0] for i in range(0, len(points))]
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    axis = create_axis(axis_length)
    # o3d.visualization.draw_geometries([point_cloud, axis])

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Hoge",  # ウインドウ名
        width=800,           # 幅
        height=600,          # 高さ
        left=50,             # 表示位置(左)
        top=50               # 表示位置(上)
    )
    vis.add_geometry(point_cloud)

    while True:
        vis.update_geometry(point_cloud)
        vis.poll_events()
        vis.update_renderer()

# draw_landmark_2d_with_index(first_landmark, flu.LIPSYNC_KEYPOINT_IDS)