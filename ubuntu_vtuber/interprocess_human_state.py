import zmq
import sys
import json

import cv2

context = zmq.Context()
subscriber = context.socket(zmq.SUB)
subscriber.connect("tcp://localhost:5555")

publisher = context.socket(zmq.PUB)
publisher.bind("tcp://127.0.0.1:9998")

import enum

class ConsumerState(enum.Enum):
    Waiting = 0
    Consuming = 1

class BaseConsumer():
    def __init__(self, subscriber, topic_name):
        self.subscriber = subscriber
        self.topic_name = topic_name
        self.mode = ConsumerState.Waiting
        self.subscriber.setsockopt(zmq.SUBSCRIBE, topic_name.encode())

    def consume(self, message):
        # print(message)
        if self.mode == ConsumerState.Waiting and message == self.topic_name:
            self.mode = ConsumerState.Consuming
        elif self.mode == ConsumerState.Consuming:
            data = json.loads(message)
            if len(data) > 0:
                self.process_data(data)
            self.mode = ConsumerState.Waiting

import datetime
import os

class ToJsonConsumer(BaseConsumer):
    def __init__(self, subscriber, topic_name, folder_name=None):
        super().__init__(subscriber, topic_name)
        self.accumulated_data = []
        if folder_name is None:
            self.folder_name = "tmp"
        else:
            self.folder_name = folder_name #  + datetime.datetime.now().strftime("%Y%m%d")
        self.current_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%m%S")

    def process_data(self, data):
        self.accumulated_data.append(data)

    def finalize(self):
        file_path = f"{self.topic_name}_{self.current_datetime}.json"
        if self.folder_name is not None:
            os.makedirs(self.folder_name, exist_ok=True)
            file_path = os.path.join(self.folder_name, file_path)

        json.dump(self.accumulated_data, open(file_path, "w"))

import open3d
import numpy as np

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
    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(points),
        lines=open3d.utility.Vector2iVector(lines),
    )
    line_set.colors = open3d.utility.Vector3dVector(colors)
    return line_set


def visualize_hand_landmark_frame(face_landmark_frame):
    if len(face_landmark_frame) == 0:
        return []
    
    point_clouds = []
    for face_landmark in face_landmark_frame:
        try:
            first_landmark = np.array(face_landmark['landmarks'])
        except:
            import IPython; IPython.embed()

        first_landmark[:, 0] = first_landmark[:, 0] * 640.0
        first_landmark[:, 1] = -first_landmark[:, 1] * 480.0

        axis_length = max(first_landmark[:, 0]) - min(first_landmark[:, 0])

        points = open3d.utility.Vector3dVector(first_landmark)
        point_cloud = open3d.geometry.PointCloud(points)
        colors = [[float(i) / len(points), 0, 0] for i in range(0, len(points))]
        point_cloud.colors = open3d.utility.Vector3dVector(colors)
        point_clouds.append(point_cloud)

    return point_clouds

# def visualize_face_landmark_frame(face_landmark_frame):
def empty(data):
    return []

FACE_LEFT_ID = 323
FACE_RIGHT_ID = 132

def to_appropriate_scale(landmark, i):
    landmarks = np.array(landmark[i]["landmarks"])
    landmarks[:, 0] = landmarks[:, 0] * 640.0
    landmarks[:, 1] = -landmarks[:, 1] * 480.0

    return landmarks

def to_appropriate_scale2(landmark):
    landmark = np.array(landmark)
    landmark[:, 0] = landmark[:, 0] * 640.0
    landmark[:, 1] = -landmark[:, 1] * 480.0

    return landmark

def get_face_center(face_data):
    center = (face_data[FACE_LEFT_ID, :] + face_data[FACE_RIGHT_ID, :]) / 2.0

    return center

SCALE = 1000

def get_wrist_position(all_data, i):
    data = all_data["HandLandmarks"]
    landmarks = to_appropriate_scale(data, i)

    wrist_point = landmarks[0, :]
    wrist_point -= get_face_center(
        to_appropriate_scale(all_data["FaceLandmarks"], 0))
    wrist_point /= SCALE

    return wrist_point

from collections import defaultdict
class HandProecssor:
    def create_publish_data(self, all_data):
        if 'FaceLandmarks' not in all_data:
            return {}

        right = None
        left = None
        indices = {}

        hand_landmarks = all_data["HandLandmarks"]

        poss = []
        for i in range(0, len(hand_landmarks)):
            write_pos = get_wrist_position(all_data, i)
            # write_pos[2] = 0.007156017295546874

            if write_pos[0] > 0:
                right = write_pos
                indices["right"] = i
            else:
                left = write_pos
                indices["left"] = i

        # import IPython; IPython.embed()

        hand_status = defaultdict(dict)
        for tag in ["right", "left"]:
            if tag in indices:
                for finger_name in flu.FINGER_IDS.keys():
                    rotations = flu.get_relative_angles_from_wrist_v3(
                        to_appropriate_scale2(
                            hand_landmarks[indices[tag]]["landmarks"]), 
                            flu.FINGER_IDS[finger_name])
                    hand_status[tag][finger_name] = [{
                        "axis": axis.tolist(),
                        "angle": angle,
                    } for axis, angle in rotations]
            else:
                hand_status[tag] = None

        return {
            "wrist_point_left": right.tolist() if right is not None else None,
            "wrist_point_right": left.tolist() if left is not None else None,
            "right": hand_status["right"],
            "left": hand_status["left"]
        }

    def create_geometries(self, data, publish_data):
        geoms = visualize_hand_landmark_frame(data)
        # points = open3d.utility.Vector3dVector([publish_data["head_position"]])
        # import IPython; IPython.embed()
        if "wrist_point_right" in publish_data and publish_data["wrist_point_right"] is not None:
            mesh_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=10.0)
            mesh_sphere.compute_vertex_normals()
            # mesh_sphere.origin = publish_data["head_position"]
            # print(publish_data["wrist_point"])
            mesh_sphere.translate(publish_data["wrist_point_right"])
            mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
            return geoms + [mesh_sphere]
        else:
            return geoms

import facelandmark_utils as flu
import pickle

CLASS_ID_TO_CLASS = ["a", "i", "u", "e", "o", "neautral"]

class FaceProcessor:
    def create_publish_data(self, all_data):
        data = all_data["FaceLandmarks"]
        landmarks = np.array(data[0]["landmarks"])
        landmarks[:, 0] = landmarks[:, 0] * 640.0
        landmarks[:, 1] = -landmarks[:, 1] * 480.0

        flu.normalize_lip(landmarks)
        feature = flu.get_lipsync_feature_v1(landmarks)
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

        head_direction = flu.simple_face_direction(landmarks, only_direction=True)
        head_direction[0] *= -1
        head_direction[1] *= -1
        head_direction[2] *= -1

        origin = np.array([ 353.02190781, -303.41849327,   46.19407082])
        center = (landmarks[FACE_LEFT_ID, :] + landmarks[FACE_RIGHT_ID, :]) / 2.0 - origin
        center /= SCALE

        # import IPython; IPython.embed()
        return {
            "head_position": center.tolist(),
            "head_direction": head_direction.tolist(),
            "lipsync_scores": class_map,
            "max_lip_shape": max_class
        }

    def create_geometries(self, data, publish_data):
        geoms = visualize_hand_landmark_frame(data)
        # points = open3d.utility.Vector3dVector([publish_data["head_position"]])
        # import IPython; IPython.embed()
        mesh_sphere = open3d.geometry.TriangleMesh.create_sphere(radius=10.0)
        mesh_sphere.compute_vertex_normals()
        # mesh_sphere.origin = publish_data["head_position"]
        # print(publish_data["head_position"])
        mesh_sphere.translate(publish_data["head_position"])
        mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
        return geoms + [mesh_sphere]

def get_geom_craetor(topic_name):
    if topic_name == "HandLandmarks":
        return HandProecssor()
    elif topic_name == "FaceLandmarks":
        return FaceProcessor()

import time
import msgpack

class HumanStateConsumer:
    def __init__(self, subscriber, publisher, logging_name=None):
        topic_names = ["HandLandmarks", "FaceLandmarks"] #, "Detection"]
        self.topic_consumers = {}
        self.processors = {}
        self.publisher = publisher

        for topic_name in topic_names:
            self.topic_consumers[topic_name] = \
                ToJsonConsumer(subscriber, topic_name, None if logging_name is None else os.path.join("logs", logging_name))
            self.processors[topic_name] = get_geom_craetor(topic_name)

        self.visualizer = open3d.visualization.Visualizer()
        self.initialized = False
        self.previous_time = time.time()
        self.folder_name = None
        if logging_name is not None:
            self.folder_name = os.path.join("logs", logging_name)
            self.publish_data = []
            self.current_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%m%S")

    def consume(self, message):
        for consumer in self.topic_consumers.values():
            consumer.consume(topic)

        all_data = {}
        for topic_name, consumer in self.topic_consumers.items():
            if len(consumer.accumulated_data) > 0:
                all_data[topic_name] = consumer.accumulated_data[-1]

        publish_data = {}
        for topic_name, consumer in self.topic_consumers.items():
            if len(consumer.accumulated_data) > 0:
                publish_data[topic_name] = \
                    self.processors[topic_name].create_publish_data(all_data)

        # print(f"Publishing: {publish_data}")
        if self.folder_name is not None:
            self.publish_data.append(publish_data)

        data = ["HumanState".encode('utf-8'), msgpack.packb(publish_data, use_bin_type=True)]
        self.publisher.send_multipart(data)

        geometries = []
        for topic_name, consumer in self.topic_consumers.items():
            if len(consumer.accumulated_data) > 0:
                geometries += \
                    self.processors[topic_name].create_geometries(
                        consumer.accumulated_data[-1], publish_data[topic_name])

        if self.initialized:
            if (time.time() - self.previous_time) > (1.0/60.0):
                # print(geometries)
                axis = create_axis(200)
                self.visualizer.clear_geometries()
                for geometry in geometries + [axis]:
                    self.visualizer.add_geometry(geometry)
                self.visualizer.poll_events()
                self.visualizer.update_renderer()
                self.previous_time = time.time()
        else:
            self.visualizer.create_window(
                window_name="Hoge",  # ウインドウ名
                width=800,           # 幅
                height=600,          # 高さ
                left=50,             # 表示位置(左)
                top=50               # 表示位置(上)
            )
            # for geometry in geometries:
            #     self.visualizer.add_geometry(geometry)
            self.initialized = True

    def finalize(self):
        for consumer in self.topic_consumers.values():
            consumer.finalize()

        print("finalized")
        # import IPython; IPython.embed()
        
        if self.folder_name is not None:
            print("dumped")
            path = os.path.join(self.folder_name, f"PublishedData_{self.current_datetime}.json")
            json.dump(self.publish_data, open(path, 'w'))

        self.visualizer.destroy_window()

# topic_names = ["HandLandmarks", "FaceLandmarks", "Detection"]
# consumers = [ToJsonConsumer(subscriber, topic_name) for topic_name in topic_names]

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--logging-name')
args = parser.parse_args()

consumers = [HumanStateConsumer(subscriber, publisher, logging_name=args.logging_name)]

try:
    while True:
        topic = subscriber.recv()
        topic = topic.decode('utf-8')

        for consumer in consumers:
            consumer.consume(topic)
except:
    import traceback
    traceback.print_exc()
finally:
    for consumer in consumers:
        consumer.finalize()
