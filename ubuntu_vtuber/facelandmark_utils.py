import numpy as np

INDEX_FINGER_IDS = [
    17, 18, 19, 20
]

INNER_LIP_IDS = list(set([
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    415, 310, 311, 312, 13, 82, 81, 80, 191,
]))
LIPSYNC_KEYPOINT_IDS = list(set([
61,
146,
91,
181,
84,
17,
314,
405,
321,
375,
291,
61,
185,
40,
39,
37,
0,
267,
269,
270,
409,
291,
78,
95,
88,
178,
87,
14,
317,
402,
318,
324,
308,
78,
191,
80,
81,
82,
13,
312,
311,
310,
415,
308,
]))

import json

def get_landmark_from_face_frames_file(filepath):
    face_landmark_frames = json.load(open(filepath, "r"))

    def __get_landmark(face_landmark_frame):
        if len(face_landmark_frame) == 0:
            return []

        return np.array(face_landmark_frame[0]['landmarks'])

    return [__get_landmark(face_landmark_frame) for face_landmark_frame in face_landmark_frames]


def simple_face_direction(landmark, only_direction=False):
    direction = np.cross(landmark[323] - landmark[93], landmark[234] - landmark[93])
    if only_direction:
        return direction / np.linalg.norm(direction)
    else:
        return np.array([
            landmark[93],
            direction / np.linalg.norm(direction)
        ])

from shapely.geometry import MultiPoint

def get_lipsync_feature_v1(landmark):
    lipsync_points = landmark[LIPSYNC_KEYPOINT_IDS]
    inner_lip_points = landmark[INNER_LIP_IDS]

    # import IPython; IPython.embed()

    direction = simple_face_direction(landmark, only_direction=True)
    
    multi_point = MultiPoint(lipsync_points[:,[0,1]])
    multi_point_bound = multi_point.bounds
    width = multi_point_bound[2] - multi_point_bound[0]
    height = multi_point_bound[3] - multi_point_bound[1]
    print(multi_point_bound)
    print(multi_point.convex_hull.area)

    # import IPython; IPython.embed()

    baisc_length = np.linalg.norm(landmark[234] - landmark[454])
    # print(baisc_length)
    # return [width / baisc_length, height / baisc_length]
    return [width, height] # , multi_point.convex_hull.area]
    # return [1.0, aspect_rate]

def get_lipsync_scores_testing(landmark):
    return ["a"]