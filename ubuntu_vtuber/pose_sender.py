import zmq
import cv2
import msgpack
from scipy.spatial.transform import Rotation
import numpy as np
import time

context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:9997")

times = []

try:
    while True:
        start = time.time()
        human_state = {}
        socket.send(msgpack.packb(human_state, use_bin_type=True))
        message = socket.recv()
        end = time.time()
        times.append((end - start) * 1000 * 1000)

        if cv2.waitKey(10) > 0:
            break
finally:
    import json
    json.dump(times, open("test.json", "w"))