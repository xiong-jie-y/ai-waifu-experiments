import zmq
import cv2
import msgpack
from scipy.spatial.transform import Rotation
import numpy as np
import json
    
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:30000")

import tempfile
import os

with tempfile.TemporaryDirectory() as tempdir:
    path = os.path.join(tempdir, "received_data.txt")
    recieved_data = []
    need_to_send = False
    try:
        while True:
            message = socket.recv()
            need_to_send = True
            
            unpacked = msgpack.unpackb(message, raw=False)
            recieved_data.append(unpacked)
            
            socket.send(msgpack.packb({}, use_bin_type=True))
            need_to_send = False

    except:
        import traceback
        traceback.print_exc()
    finally:
        json.dump(recieved_data, open(path, "w"))
        print(f"Dumped to {path}")
        import IPython; IPython.embed()
        
        socket.send(msgpack.packb({}, use_bin_type=True))