import zmq
import json

import argparse
import os
import datetime

import detectors.skinship as dss

def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:10002")

    detector = dss.GoingInsideDetector()
    
    while True:
        print("start")
        msg = socket.recv()
        print(msg)
        sensor_val = json.loads(msg.decode('utf-8'))
        # detector.add(dss.from_dict_exp_to_array(sensor_val))
        for user_accceleration in sensor_val["userAcceleration"]:
            detector.add(user_accceleration)
            # state = detector.get_state()
            # if state == "going_inside":
            #     break   

        state = detector.get_state()
        print(state)
        socket.send(state.encode('utf-8'))

if __name__ == "__main__":
    main()
