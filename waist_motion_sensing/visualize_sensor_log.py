import json
import numpy as np

import argparse

import streamlit as st

import matplotlib.pyplot as plt
import plotly.graph_objects as go

import os

import numpy as np
from scipy import signal

def to_lower_dim(timestamped_values):
    delta_ts = timestamped_values[:, 0][1:] \
                 - timestamped_values[:, 0][0:-1]
    vs = timestamped_values[1:, 1:] / np.expand_dims(delta_ts, 1)
    return np.concatenate((
        np.expand_dims(timestamped_values[1:, 0], 1), vs), axis=1)

def norm_series(xyzt):
    acc_norm = np.linalg.norm(xyzt[:, 1:4], axis=1)
    tss = xyzt[:, 0]
    return np.concatenate((np.expand_dims(tss, 1), np.expand_dims(acc_norm, 1)), axis=1)

def linear_base_lowpass(xs, factor):
    previous_value = 0
    new_values = []

    for x in xs:
        previous_value = (x - previous_value) * factor + previous_value
        new_values.append(previous_value)

    return np.array(new_values)


def draw_fft(timeseries):
    import plotly.express as px
    fft = np.fft.fft(timeseries)
    fig2 = px.line(y=[np.sqrt(c.real ** 2 + c.imag ** 2) for c in fft])
    # st.write([np.sqrt(c.real ** 2 + c.imag ** 2) for c in fft])
    st.write(fig2)    

def analyze_frequency(timeseries):
    FRAME_PER_SEC = 60
    window_period_s = st.slider("window_period_s", 1, len(timeseries) // 60)
    window_size = window_period_s * FRAME_PER_SEC
    window_func = st.radio("Window", [np.hamming, np.hanning, np.blackman, np.bartlett])    

    windows = []
    for i in range(0, len(timeseries) - window_size):
        DFT = np.fft.fft(window_func(window_size) * timeseries[i:i + window_size])
        DFT = DFT[:len(DFT) // 2]
        windows.append([np.sqrt(c.real ** 2 + c.imag ** 2) for c in DFT])
    windows = np.array(windows)

    fig = go.Figure(data=go.Heatmap(
                    z=windows))

    fig.update_layout(
        xaxis_title="Frequency [Hz]",
        yaxis_title="Frame No",)
    st.write(fig)

    draw_fft(timeseries)


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('input_file')
    # args = parser.parse_args()
    log_dir = "logs"
    files = os.listdir(log_dir)
    file_path = st.sidebar.selectbox('choose your log file', files)
    sensor_log = json.load(open(os.path.join(log_dir, file_path), "r"))["data"]
    length = 0
    for key, timestamped_data in sensor_log.items():
        sensor_log[key] = np.array(timestamped_data)
        length = len(sensor_log[key])

    visualize_term = int(st.sidebar.text_input('text', 300))
    visualize_index = st.sidebar.slider("aa", 0, length // visualize_term, 1)

    for key in sensor_log:
        sensor_log[key] = sensor_log[key][visualize_index * visualize_term: (visualize_index + 1) * visualize_term]

    import utils.timestamped_sequence as uts
    import time
    sensor_log["userAccelerationNorm"] = norm_series(sensor_log["userAcceleration"])
    sensor_log["filteredUserAcceleration"] = \
        uts.get_lowpass_timestamped_sequence(sensor_log["userAcceleration"])

    analyze_frequency(sensor_log["filteredUserAcceleration"][:, 2])

    import detectors.skinship as dss
    detector = dss.GoingInsideDetector()
    for state in sensor_log["userAcceleration"][:500]:
        detector.add(state)
    st.markdown(detector.get_state())

    facotr = 1 / 3
    lowpass_x2 = linear_base_lowpass(sensor_log["userAcceleration"][:,1], facotr)
    lowpass_y2 = linear_base_lowpass(sensor_log["userAcceleration"][:,2], facotr)
    lowpass_z2 = linear_base_lowpass(sensor_log["userAcceleration"][:,3], facotr)
    sensor_log["filteredUserAcceleration2"] = np.concatenate(
        (np.expand_dims(sensor_log["userAcceleration"][:,0], 1), 
        np.expand_dims(lowpass_x2, 1), 
        np.expand_dims(lowpass_y2, 1),
        np.expand_dims(lowpass_z2, 1)), axis=1)
    # delta_ts = sensor_log["userAcceleration"][:, 0][1:] \
    #              - sensor_log["userAcceleration"][:, 0][0:-1]
    # vs = sensor_log["userAcceleration"][1:, 1:] / np.expand_dims(delta_ts, 1)
    # sensor_log["userVelocity"] = np.concatenate((
    #     np.expand_dims(sensor_log["userAcceleration"][1:, 0], 1), vs), axis=1)
    source_acc = st.selectbox("source acc", ["userAcceleration", "filteredUserAcceleration"])
    sensor_log["userVelocity"] = to_lower_dim(sensor_log[source_acc])
    sensor_log["userPosition"] = to_lower_dim(sensor_log["userVelocity"])
    sensor_log["userPositionNorm"] = norm_series(sensor_log["userPosition"])
    # import IPython; IPython.embed()

    # import IPython; IPython.embed()

    # ACCELERATION_VALUES = ["acceleration_raw", "acceleration", "gravity", "userAcceleration"]
    ACCELERATION_VALUES = st.sidebar.multiselect(
        "aa", 
        ["acceleration", "gravity", "userAcceleration", 
        "userAccelerationNorm", "filteredUserAcceleration",
        "filteredUserAcceleration2"
        ], 
        default=["userAcceleration", "filteredUserAcceleration", "filteredUserAcceleration2"])
    VELOCITY_VALUES = ["userVelocity"]
    POSITION_VALUES = ["userPosition", "userPositionNorm", "slamBasedDevicePosition"]
    ROTATION_VALUES = ["rotationRate", "rotationRateUnbiased"]
    POSE = ["attitude"]
    for sensor_value_group in [ACCELERATION_VALUES, VELOCITY_VALUES, POSITION_VALUES, ROTATION_VALUES, POSE]:
        fig = go.Figure()
        for sensor_value_name in sensor_value_group:
            # [visualize_index * visualize_term: (visualize_index + 1) * visualize_term]
            acc = sensor_log[sensor_value_name]
            if len(acc[0]) == 4:
                labels = "txyz"
                for i in [1,2,3]:
                    fig.add_trace(go.Scatter(x=acc[:,0], y=acc[:,i],
                                    mode='lines+markers',
                                    name=f"{sensor_value_name}_{labels[i]}"))
            elif len(acc[0]) == 5:
                labels = "txyzw"
                for i in [1,2,3, 4]:
                    fig.add_trace(go.Scatter(x=acc[:,0], y=acc[:,i],
                                    mode='lines+markers',
                                    name=f"{sensor_value_name}_{labels[i]}"))
            elif len(acc[0]) == 2:
                fig.add_trace(go.Scatter(x=acc[:,0], y=acc[:, 1],
                                mode='lines+markers',
                                name=f"{sensor_value_name}"))
                # import IPython; IPython.embed()
                
            # plt.plot(acc[:,0], acc[:,1], label=f"{sensor_value_name}_x")
            # plt.plot(acc[:,0], acc[:,2], label=f"{sensor_value_name}_y")
            # plt.plot(acc[:,0], acc[:,3], label=f"{sensor_value_name}_z")
            # plt.legend()
        st.write(fig)
        # plt.show()

if __name__ == "__main__":
    main()