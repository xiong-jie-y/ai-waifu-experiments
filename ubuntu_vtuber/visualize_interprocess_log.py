import open3d as o3d

import numpy as np
import json

def create_axis(length=1.0):
    points = [
        [0, 0, 0],
        [length, 0, 0],
        [0, length, 0],
        [0, 0, length]
    ]
    lines = [
        [0, 1],
        [0, 2],
        [0, 3]
    ]
    colors = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

def get_first_landmarks_in_frames(filepath):
    face_landmark_frames = json.load(open(filepath, "r"))

    # import IPython; IPython.embed()
    # def __get_landmark(face_landmark_frame):
    #     hands = face_landmark_frame['hands']

    #     return np.array(hands[0]['landmarks'])
    def __get_landmark(face_landmark_frame):
        if len(face_landmark_frame) == 0:
            return []
        
        return np.array(face_landmark_frame[0]['landmarks'])

    return [__get_landmark(face_landmark_frame) for face_landmark_frame in face_landmark_frames]

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--filepath')
args = parser.parse_args()

first_landmarks = get_first_landmarks_in_frames(args.filepath)
first_landmark = first_landmarks[2]
# first_landmark = first_landmarks[0]

print(first_landmark)

# import IPython; IPython.embed()

# first_landmark[:, 0] = first_landmark[:, 0] * 256.0
# first_landmark[:, 1] = first_landmark[:, 1] * 256.0

first_landmark[:, 0] = first_landmark[:, 0] * 640.0
first_landmark[:, 1] = -first_landmark[:, 1] * 480.0

axis_length = max(first_landmark[:, 0]) - min(first_landmark[:, 0])

# min_z = min(first_landmark[:, 2])
# max_z = max(first_landmark[:, 2])
# first_landmark[:, 2] = (first_landmark[:, 2]) / 255
# first_landmark[:, 2] = (first_landmark[:, 2] - min_z) / (max_z - min_z)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import facelandmark_utils as flu

# flu.get_lip_bounding_box(first_landmark)

def draw_landmark_2d_with_index(landmark, filter_ids=[]):
    fig = plt.figure()
    ax1 = fig.add_subplot(111 , projection='3d')

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
    plt.show()

draw_landmark_2d_with_index(first_landmark)

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