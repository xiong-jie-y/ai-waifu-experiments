from scipy.spatial.transform import Rotation
import pickle
from shapely.geometry import MultiPoint
import streamlit as st
import json
import numpy as np

WRIST_IDS = [0]

FINGER_IDS = dict(
    index_finger=[
        5, 6, 7, 8
    ],
    middle_finger=[
        9, 10, 11, 12
    ],
    ring_finger=[
        13, 14, 15, 16
    ],
    pinky_finger=[
        17, 18, 19, 20
    ],
    thumb_finger=[
        1, 2, 3, 4
    ]
)

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


def get_landmark_from_face_frames_file(filepath, denormalize=True):
    face_landmark_frames = json.load(open(filepath, "r"))

    def __get_landmark(face_landmark_frame):
        if len(face_landmark_frame) == 0:
            return []

        landmark = np.array(face_landmark_frame[0]['landmarks'])
        if denormalize:
            landmark[:, 0] = landmark[:, 0] * 640.0
            landmark[:, 1] = -landmark[:, 1] * 480.0
        return landmark

    return [__get_landmark(face_landmark_frame) for face_landmark_frame in face_landmark_frames]


def simple_face_direction(landmark, only_direction=False):
    direction = np.cross(landmark[323] - landmark[93],
                         landmark[234] - landmark[93])
    if only_direction:
        return direction / np.linalg.norm(direction)
    else:
        return np.array([
            landmark[93],
            direction / np.linalg.norm(direction)
        ])


def get_lipsync_feature_v1(landmark):
    lipsync_points = landmark[LIPSYNC_KEYPOINT_IDS]
    inner_lip_points = landmark[INNER_LIP_IDS]

    # import IPython; IPython.embed()

    direction = simple_face_direction(landmark, only_direction=True)

    multi_point = MultiPoint(lipsync_points[:, [0, 1]])
    multi_point_bound = multi_point.bounds
    width = multi_point_bound[2] - multi_point_bound[0]
    height = multi_point_bound[3] - multi_point_bound[1]
    # print(multi_point_bound)
    # print(multi_point.convex_hull.area)

    # import IPython; IPython.embed()

    baisc_length = np.linalg.norm(landmark[234] - landmark[454])
    # print(baisc_length)
    # return [width / baisc_length, height / baisc_length]
    return [width, height]  # , multi_point.convex_hull.area]
    # return [1.0, aspect_rate]


def get_lipsync_scores_testing(landmark):
    return ["a"]


CLASS_ID_TO_CLASS = ["a", "i", "u", "e", "o", "neautral"]


def get_lipclass_shape(landmark, should_normalize_lip=True):
    if should_normalize_lip:
        normalize_lip(landmark)
    feature = get_lipsync_feature_v1(landmark)

    model = pickle.load(open("mouth_classification_model.pkl", "rb"))
    proba = model.predict_proba([feature])[0]
    class_map = {}
    max_class = -1
    max_score = -1

    for i, p_i in enumerate(proba):
        class_map[CLASS_ID_TO_CLASS[i]] = p_i
        if p_i > max_score:
            max_score = p_i
            max_class = CLASS_ID_TO_CLASS[i]

    return class_map

from numba import jit

def get_shortest_rotvec_between_two_vector(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    rotation_axis = np.cross(a, b)

    # Because they are unit vectors.
    theta = np.arccos(a.dot(b))

    return rotation_axis, theta


@jit(nopython=True)
def get_shortest_rotvec_between_two_vector2(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    rotation_axis = np.cross(a, b)

    # Because they are unit vectors.
    theta = np.arccos(a.dot(b))

    return rotation_axis, theta

# @jit(nopython=True)
def get_shortest_rotvec_between_two_vector3(a, b):
    # a = a / np.linalg.norm(a, axis=1)
    # b = b / np.linalg.norm(b, axis=1)
    a = a / np.sum(a * a, axis=1)
    b = b / np.sum(b * b, axis=1)

    rotation_axis = np.cross(a, b, axis=1)

    # Because they are unit vectors.
    # theta = np.arccos([np.dot(a_i,b_i) for a_i,b_i in zip(a, b)])
    theta = np.arccos(np.sum(a * b, axis=1))

    return rotation_axis, theta


def normalize_lip(landmark):
    direction = landmark[291] - landmark[61]
    base = np.array([1, 0, 0])
    rotation_axis, theta = get_shortest_rotvec_between_two_vector(
        base, direction)
    rotvec = Rotation.from_rotvec(rotation_axis * theta)

    # st.write(theta)
    # st.write(rotation_axis)

    # import IPython; IPython.embed()
    landmark[LIPSYNC_KEYPOINT_IDS] = rotvec.apply(
        landmark[LIPSYNC_KEYPOINT_IDS])

@jit
def get_relative_angles_from_wrist_v2(landmark, finger_ids):
    # import IPython; IPython.embed()
    
    finger_pos = landmark[WRIST_IDS + finger_ids]
    finger_diff = finger_pos[1:] - finger_pos[:-1]

    # rotations = []
    # for a, b in zip(finger_diff[1:], finger_diff[:-1]):
    #     rotations.append(get_shortest_rotvec_between_two_vector2(a, b))
    return get_shortest_rotvec_between_two_vector3(finger_diff[1:], finger_diff[:-1])

    # return rotations

@jit(nopython=True)
def get_relative_angles(position_array):
    finger_diff = position_array[1:] - position_array[:-1]

    # rotations = []
    # for a, b in zip(finger_diff[1:], finger_diff[:-1]):
    #     rotations.append(get_shortest_rotvec_between_two_vector2(a, b))

    return [
        get_shortest_rotvec_between_two_vector2(a, b) 
        for a,b in zip(finger_diff[1:], finger_diff[:-1])
    ]

def get_relative_angles_no_numba(position_array):
    finger_diff = position_array[1:] - position_array[:-1]

    # rotations = []
    # for a, b in zip(finger_diff[1:], finger_diff[:-1]):
    #     rotations.append(get_shortest_rotvec_between_two_vector2(a, b))

    return [
        get_shortest_rotvec_between_two_vector(a, b) 
        for a,b in zip(finger_diff[1:], finger_diff[:-1])
    ]

def get_relative_angles_from_wrist_v3(landmark, finger_ids):
    # finger_pos = landmark[WRIST_IDS + finger_ids]
    return get_relative_angles(landmark[WRIST_IDS + finger_ids])

def get_relative_angles_from_wrist_v4(landmark, finger_ids):
    # finger_pos = landmark[WRIST_IDS + finger_ids]
    return get_relative_angles_no_numba(landmark[WRIST_IDS + finger_ids])

def get_relative_angles_from_wrist_v1(landmark, finger_ids):
    finger_pos = landmark[WRIST_IDS + finger_ids]
    finger_diff = finger_pos[1:] - finger_pos[:-1]

    # rotations = []
    # for a, b in zip(finger_diff[1:], finger_diff[:-1]):
    #     rotations.append(get_shortest_rotvec_between_two_vector2(a, b))

    return [
        get_shortest_rotvec_between_two_vector2(a, b) 
        for a,b in zip(finger_diff[1:], finger_diff[:-1])
    ]