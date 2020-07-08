#%%

# %%
import streamlit as st
import result

def detection_analysis_dashboard():
    @st.cache(allow_output_mutation=True)
    def load_data():
        # return result.DataWithTimestamp.load_from_path("data/detection_result_20200706_211314")
        # return result.DataWithTimestamp.load_from_path("data/detection_result_20200707_222602")
        return result.DataWithTimestamp.load_from_path("data/detection_result_20200707_225648_real")
    logged_data = load_data()

    def get_from_box(box):
        return ((box[2] + box[0])/2 - 250)/50 * 0.05

    def get_x_from_box(box):
        return ((box[3] + box[1])/2 - 425)

    def get_positions(detection_lists):
        positions = []
        for detection_list in detection_lists:
            found = False
            for detection in detection_list[1]:
                if detection["label"] == "onahole":
                    found = True
                    positions.append(get_x_from_box(detection["box"]))
                    break
                    
            if not found:
                if len(positions) == 0:
                    positions.append(None)
                else:
                    positions.append(positions[-1])

        return positions



    # %%
    images = logged_data.get_images_with_timestamp('image_with_detection')
    detection_lists = logged_data.get_data("detections")

    st.write(f"{len(images)}: {len(detection_lists)}")

    frame_no = st.slider("Frame no", 0, len(images))
    selected_timestamp = images[frame_no][0]
    st.image(images[frame_no][1])
    # st.write(detection_lists[frame_no])

    import plotly.graph_objects as go
    timestamps = [detection_list[0] for detection_list in detection_lists]
    y_positions = get_positions(detection_lists)
    fig = go.Figure(
        data=[
            go.Scatter(x=timestamps, y=y_positions),
            go.Scatter(x=[selected_timestamp, selected_timestamp], y=[-0.2,0.2])],
        layout_title_text="A Figure Displayed with fig.show()"
    )
    st.write(fig)

    return logged_data, frame_no

    # %%
detection_analysis_dashboard()

# %%
