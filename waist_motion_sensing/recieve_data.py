import zmq
import json

import argparse
import os
import datetime

def main():
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:10001")

    while True:
        print("start")
        msg = socket.recv()
        data = json.loads(msg.decode('utf-8'))
        current_datetime = datetime.datetime.now().strftime("%Y%m%d-%H%m%S")
        os.makedirs("logs", exist_ok=True)
        log_name = data["log_name"]
        json.dump(data, open(f"logs/{log_name}_{current_datetime}.json", "w"))
        print(f"recv {msg}")
        socket.send(b'test')

if __name__ == "__main__":
    main()

