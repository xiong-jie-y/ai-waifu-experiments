import cv2
import time

import collections

import matplotlib.pyplot as plt

import face_alignment
from skimage import io

import zmq
import msgpack

import argparse
import streamlit as st

def grab_frame(cap):
    ret,frame = cap.read()
    return cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

class OriginalExtractor:
    def __init__(self):
        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False)
        pred_type = collections.namedtuple('prediction_type', ['slice', 'color'])
        self.pred_types = {'face': pred_type(slice(0, 17), (0.682, 0.780, 0.909, 0.5)),
                    'eyebrow1': pred_type(slice(17, 22), (1.0, 0.498, 0.055, 0.4)),
                    'eyebrow2': pred_type(slice(22, 27), (1.0, 0.498, 0.055, 0.4)),
                    'nose': pred_type(slice(27, 31), (0.345, 0.239, 0.443, 0.4)),
                    'nostril': pred_type(slice(31, 36), (0.345, 0.239, 0.443, 0.4)),
                    'eye1': pred_type(slice(36, 42), (0.596, 0.875, 0.541, 0.3)),
                    'eye2': pred_type(slice(42, 48), (0.596, 0.875, 0.541, 0.3)),
                    'lips': pred_type(slice(48, 60), (0.596, 0.875, 0.541, 0.3)),
                    'teeth': pred_type(slice(60, 68), (0.596, 0.875, 0.541, 0.4))
                    }
        self.plot_style = dict(marker='o',
                  markersize=4,
                  linestyle='-',
                  lw=2)
        self.init_figure = False

    def __maybe_init_figure(self):
        if not self.init_figure:
            self.fig = plt.figure(figsize=plt.figaspect(.5))
            self.ax = self.fig.add_subplot(1, 1, 1)
            self.im_ax = self.ax.imshow(grab_frame(capture))
            self.lines = [self.ax.plot([],[], color=pred_type.color, **self.plot_style)[0] for pred_type in self.pred_types.values()]
            plt.ion()
            self.init_figure = True

    def get_state(self, image_rgb, visualize=False):
        self.__maybe_init_figure()

        landmarks = self.fa.get_landmarks(img_rgb)
        if landmarks is not None:
            preds = landmarks[-1]
            if visualize:
                # plt.clf()
                # ax.imshow(img_rgb)
                self.im_ax.set_data(img_rgb)

                for i, pred_type in enumerate(self.pred_types.values()):
                    self.lines[i].set_data(preds[pred_type.slice, 0],
                            preds[pred_type.slice, 1])
                    plt.pause(0.001)
                # plt.draw()
                # fig.canvas.draw()
            eye_points = preds[self.pred_types['eye1'].slice]
            print(eye_points)
            assert len(eye_points) == 6

            ey = eye_points[:, 1].tolist()
            max_y = max(ey)
            min_y = min(ey)
            print(max_y, min_y)
            face_state = dict(
                eye_open_degree=(max_y - min_y)
            )
            print(face_state)
            return face_state
        else:
            if visualize:
                # plt.clf()
                self.im_ax.set_data(img_rgb)
                    # fig.canvas.draw()
                plt.pause(0.001)
            return None

from scipy.spatial import distance as dist
import imutils
from imutils import face_utils
import dlib
from collections import deque
from scipy.spatial.transform import Rotation

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
	C = dist.euclidean(eye[0], eye[3])

	# compute the eye aspect ratio
	ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
	return ear

import math

def rotation_vector_to_quaternion(pose):
    rotation_vector, translation_vector = pose
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    # import IPython; IPython.embed()

    proj_matrix = np.hstack((rvec_matrix, translation_vector.reshape(3, 1)))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

    print(eulerAngles)

    # pitch, yaw, roll =

    return Rotation.from_euler('xyz', np.array([math.radians(_) for _ in eulerAngles]))

class PoseEstimator:

    def __init__(self, img_size=(480, 640)):
        print(img_size)
        self.size = img_size

        self.model_points_68 = self._get_full_model_points()

        # Camera internals
        self.focal_length = self.size[1]
        self.camera_center = (self.size[1] / 2, self.size[0] / 2)
        self.camera_matrix = np.array(
            [[self.focal_length, 0, self.camera_center[0]],
             [0, self.focal_length, self.camera_center[1]],
             [0, 0, 1]], dtype="double")

        # Assuming no lens distortion
        self.dist_coeefs = np.zeros((4, 1))

    def _get_full_model_points(self, filename='assets/model.txt'):
        """Get all 68 3D model points from file"""
        raw_value = []
        with open(filename) as file:
            for line in file:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T

        # Transform the model into a front view.
        model_points[:, 2] *= -1

        return model_points

    def estimate_3d_pose(self, image_points):
        rvec2 = np.zeros(3)
        tvec2 = np.zeros(3)
        (_, rotation_vector, translation_vector) = cv2.solvePnP(
            self.model_points_68, np.array(image_points).astype(np.float32), self.camera_matrix, self.dist_coeefs,
            rvec=rvec2, tvec=tvec2)
            # useExtrinsicGuess=True)
        print(rotation_vector)
        print(rvec2)
        print(translation_vector)
        print(tvec2)
        # assert rvec2 == rotation_vector and tvec2 == translation_vector
        return rotation_vector_to_quaternion((rotation_vector, translation_vector))

PRESERVE_FRAME = 3

import face_recognition.api as face_api

class DlibEARBasedHumanStateEstimator:
    def __init__(self, use_cnn=True):
        print("[INFO] loading facial landmark predictor...")
        if use_cnn:
            self.detector = face_api.cnn_face_detector
        else:
            self.detector = dlib.get_frontal_face_detector()

        self.use_cnn = use_cnn

        # self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        # self.predictor = face_api.pose_predictor_68_point

        self.predictor = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

        self.init_figure = False
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        self.eye_open_degrees = deque([])
        self.pose_estimator = None

    def get_state(self, image_rgb, visualize=False):
        # frame = imutils.resize(image_rgb, width=900)
        frame = imutils.resize(image_rgb)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # gray = frame

        if self.pose_estimator is None:
            self.pose_estimator = PoseEstimator(frame.shape)

        # st.markdown('test')
        # detect faces in the grayscale frame
        if self.use_cnn:
            rects = self.detector(frame, 0)
        else:
            rects = self.detector(gray, 0)

        # loop over the face detections
        if len(rects) == 0:
            if visualize:
                cv2.imshow("Frame", frame)
                # st.image(frame)
            return None
        elif len(rects) == 1:
            # for rect in rects:
            if self.use_cnn:
                rect = rects[0].rect
            else:
                rect = rects[0]
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            # import IPython; IPython.embed()
            
            # shape = self.predictor(gray, rect)
            # shape = face_utils.shape_to_np(shape)
            # import IPython; IPython.embed()

            bd = np.array([rect.left(), rect.top(), rect.right(), rect.bottom()])
            landmarks = self.predictor.get_landmarks(gray)
            if landmarks is None:
                return

            shape = np.array(landmarks[-1])
            for landmark in shape:
                if not (0 <= landmark[0] <= 640) or not (0 <= landmark[1] < 480):
                    return

            print(shape)

            # import IPython; IPython.embed()

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            eye_open_degree = min(1.0, 1 - max((ear - 0.23)/0.1, 0))
            self.eye_open_degrees.append(eye_open_degree)
            if len(self.eye_open_degrees) > PRESERVE_FRAME:
                self.eye_open_degrees.popleft()
            mean_eod = np.mean(self.eye_open_degrees)
            # import IPython; IPython.embed()
            head_pose = self.pose_estimator.estimate_3d_pose(shape)
            head_rotation = head_pose.as_quat().tolist()

            print(head_rotation)
            if visualize:
                frame = np.zeros(frame.shape)
                # cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                # cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                for landmark_loc in shape:
                    cv2.circle(frame, tuple(landmark_loc), 1, (0,0,255), -1)

                cv2.putText(frame, "EAR: {:.2f}".format(mean_eod), (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("Frame", frame)
                # st.image(frame)
            return dict(
                eye_open_degree=mean_eod,
                head_rotation=head_rotation,
            )
        else:
            print("two face detected")

# @st.cache(hash_funcs={
#     cv2.VideoCapture: lambda _: None, 
#     zmq.sugar.context.Context: lambda _: None,
#     zmq.sugar.socket.Socket: lambda _: None,
#     })
def one_time_init():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:9998")

    # human_state_estimator = OriginalExtractor()
    human_state_estimator = DlibEARBasedHumanStateEstimator()

    capture = cv2.VideoCapture(6)

    # capture.set(3,320) # 幅
    # capture.set(4,240) # 高さ
    capture.set(5,20)  # FPS

    return human_state_estimator, capture, context, socket


parser = argparse.ArgumentParser()
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--publish', action='store_true')
args = parser.parse_args()

visualize = args.visualize
publish = args.publish

# publish = st.checkbox('Turn on Publishing')
# visualize = st.checkbox('Turn on Visualization')

human_state_estimator, capture, context, socket = one_time_init()

import numpy as np

while True:
    ret, image = capture.read()
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    human_state = human_state_estimator.get_state(img_rgb, visualize)

    if human_state and publish:
        message = socket.recv()
        print("Received request: %s" % message)
        socket.send(msgpack.packb(human_state, use_bin_type=True))

    # cv2.imshow("camera",image)
    if cv2.waitKey(10) > 0:
        break

capture.release()
# cv2.destroyAllWindows()
