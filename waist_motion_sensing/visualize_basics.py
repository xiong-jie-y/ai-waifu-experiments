import numpy as np
import streamlit as st

import plotly.express as px

def createSineWave (A, f0, fs, length):
    """振幅A、基本周波数f0、サンプリング周波数 fs、
    長さlength秒の正弦波を作成して返す"""
    data = []
    # [-1.0, 1.0]の小数値が入った波を作成
    for n in range(length * fs):  # nはサンプルインデックス
        s = A * np.sin(2 * np.pi * f0 * n / fs)
        # 振幅が大きい時はクリッピング
        if s > 1.0:  s = 1.0
        if s < -1.0: s = -1.0
        data.append(s)
    # [-32768, 32767]の整数値に変換
    data = [int(x * 32767.0) for x in data]
#    plot(data[0:100]); show()
    # バイナリに変換
    # data = struct.pack("h" * len(data), *data)  # listに*をつけると引数展開される
    return data

def draw_fft(timeseries):
    fft = np.fft.fft(timeseries)
    fig2 = px.line(y=[np.sqrt(c.real ** 2 + c.imag ** 2) for c in fft])
    # st.write([np.sqrt(c.real ** 2 + c.imag ** 2) for c in fft])
    st.write(fig2)    

data = createSineWave(1, 1, 50, 5)
fig2 = px.line(y=data)
# st.write([np.sqrt(c.real ** 2 + c.imag ** 2) for c in fft])
st.write(fig2)    
draw_fft(data)
