import streamlit as st
import cv2

@st.cache(allow_output_mutation=True)
def get_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if ret:
            frames.append(frame[:,:,::-1])
        else:
            break
    return frames

def show_video(path):
    frames = get_frames(path)
    animation_frame_id = st.slider("animation id", 0, len(frames) - 1, key=f"{path}_anim")
    st.image(frames[animation_frame_id], width=500)

def show_all_frames(path):
    frames = get_frames(path)
    for i, frame in enumerate(frames):
        st.write(i)
        st.image(frames[i], width=500)

show_video("/home/yusuke/Downloads/pixiv/クロカジ/クロカジ - ニッカノ・ハメハメ 14 (78122446) .avi")
show_all_frames("/home/yusuke/Downloads/pixiv/クロカジ/クロカジ - ニッカノ・ハメハメ 14・ドピュル (78157657) .avi")
