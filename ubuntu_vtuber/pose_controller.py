import zmq
import cv2
import msgpack
from scipy.spatial.transform import Rotation
import numpy as np
    
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:9997")

while True:
    message = socket.recv()
    # euler = np.array([ 0.5880026 ,  1.88786223, -0.5880026])
    # rot = Rotation.from_euler('XYX', euler)
    human_state = {
        # "leg_pose": rot.as_quat().tolist()
    }
    # print(f"sending {human_state}")
    socket.send(msgpack.packb(human_state, use_bin_type=True))

    if cv2.waitKey(10) > 0:
        break