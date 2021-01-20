import numpy as np
from scipy import signal
import time

import streamlit as st

fp = 3000       #通過域端周波数[Hz]
fs = 6000       #阻止域端周波数[Hz]
gpass = 3       #通過域端最大損失[dB]
gstop = 40      #阻止域端最小損失[dB]

#バターワースフィルタ（ローパス）
def lowpass(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2                           #ナイキスト周波数
    wp = fp / fn                                  #ナイキスト周波数で通過域端周波数を正規化
    ws = fs / fn                                  #ナイキスト周波数で阻止域端周波数を正規化
    N, Wn = signal.buttord(wp, ws, gpass, gstop)  #オーダーとバターワースの正規化周波数を計算
    b, a = signal.butter(N, Wn, "low")            #フィルタ伝達関数の分子と分母を計算
    y = signal.filtfilt(b, a, x)                  #信号に対してフィルタをかける
    return y                                      #フィルタ後の信号を返す

def get_lowpass_timestamped_sequence(timestamped_sequence):
    samplerate = 60
    # fp = st.slider('fp', 0, 100, 10)       #通過域端周波数[Hz]
    # fp = st.slider('fp', 0, 100, 8)       #通過域端周波数[Hz]
    # fs = st.slider('fs', 0, 100, 20)       #阻止域端周波数[Hz]
    fp = 8
    fs = 20
    gpass = 3       #通過域端最大損失[dB]
    gstop = 40      #阻止域端最小損失[dB]

    start = time.time()
    lowpass_x = lowpass(timestamped_sequence[:,1], samplerate, fp, fs, gpass, gstop)
    lowpass_y = lowpass(timestamped_sequence[:,2], samplerate, fp, fs, gpass, gstop)
    lowpass_z = lowpass(timestamped_sequence[:,3], samplerate, fp, fs, gpass, gstop)
    end = time.time()
    result = np.concatenate(
        (np.expand_dims(timestamped_sequence[:,0], 1), 
        np.expand_dims(lowpass_x, 1), 
        np.expand_dims(lowpass_y, 1),
        np.expand_dims(lowpass_z, 1)), axis=1)
    st.write(end - start)
    return result