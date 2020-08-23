import argparse
import json

import plotly.express as px

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dir')
    args = parser.parse_args()

    zs = []
    publish_data = json.load(open(args.input_dir, "r"))
    for frame_data in publish_data:
        # print(frame_data)
        if "HandLandmarks" in frame_data and "wrist_point_left" in frame_data["HandLandmarks"] \
            and frame_data["HandLandmarks"]["wrist_point_left"] is not None:
            wrist_point = frame_data["HandLandmarks"]["wrist_point_left"]
            zs.append(wrist_point[2])

    fig = px.histogram(x=zs)
    fig.show()

if __name__ == "__main__":
    main()