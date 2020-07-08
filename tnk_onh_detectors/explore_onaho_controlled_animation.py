# %%

# %%
import onaho_based_animation_controller
import streamlit as st
import explorer_2d_detection_app

logged_data, frame_no = explorer_2d_detection_app.detection_analysis_dashboard()


anim_controller = onaho_based_animation_controller.OnahoAnimationController(
    onaho_based_animation_controller.SimpleEstimator())
detection_list = logged_data.get_data("detections")[frame_no]
frame = anim_controller.consume_detection(detection_list)
st.image(frame, width=300)
