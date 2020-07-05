#%%
import slam_map
import os

label_names = [
    ("200702-204426", "onahole"),
    ("200702-205440", "onahole"),
    ("200702-210146", "onahole"),
    ("200702-210546", "onahole"),
    # ("200703-213354", "tnk"),
    ("200703-214340", "tnk"),
    ("200703-215526", "tnk"),
    ("200704-141852", "tnk"),
    ("200704-143117", "onahole"),
    ("200704-143117", "onahole")
]
bounding_box_lists = []
rgbd_images = []
for folder_name, label in label_names:
    map = slam_map.SLAMMap.load_map_from_rtab_result_dir(
        os.path.expanduser(f"~/Documents/RTAB-Map/{folder_name}")
    )
    bds = map.get_camera_centered_bounding_boxes()

    # override labels
    for bd_list in bds:
        for i in range(0, len(bd_list)):
            bd_list[i] = (label, bd_list[i][1])

    bounding_box_lists += bds
    rgbd_images += map.rgbd_images

    # map.show_2d_dataset(bounding_box_lists)

#%%


# %%
import dataset

assert(len(rgbd_images) == len(bounding_box_lists))

dataset.create_object_detection_api_tf_record(rgbd_images, bounding_box_lists, "onaho_v4")

# %%
