#%%
import open3d as o3d

import numpy as np
import json

def create_axis():
    points = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
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

# parser = argparse.ArgumentParser()
# parser.add_argument('--filepath')
# args = parser.parse_args()

filepath = "HandLandmarks_20200429-120441.json"

first_landmarks = get_first_landmarks_in_frames(filepath)
first_landmark = first_landmarks[0]

min_z = min(first_landmark[:, 2])
max_z = max(first_landmark[:, 2])
first_landmark[:, 2] = 1.0 - (first_landmark[:, 2] - min_z) / (max_z - min_z)

points = o3d.utility.Vector3dVector(first_landmark)

point_cloud = o3d.geometry.PointCloud(points)
axis = create_axis()
# o3d.visualization.draw_geometries([point_cloud, axis])

import numpy as np
import open3d as o3
from open3d import JVisualizer

visualizer = JVisualizer()
visualizer.add_geometry(point_cloud)
visualizer.show()

# %%
