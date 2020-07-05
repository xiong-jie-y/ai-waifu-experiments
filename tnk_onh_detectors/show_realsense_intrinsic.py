# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

# Create a pipeline
pipeline = rs.pipeline()

# print('project_point_to_pixel' in dir(rs))

# #Create a config and configure the pipeline to stream
# #  different resolutions of color and depth streams
# config = rs.config()
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# # Start streaming
# profile = pipeline.start(config)

config = rs.config()
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
cfg = pipeline.start(config) # Start pipeline and get the configuration it found
profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
print(intr.ppx)

print(intr)

video_profile = cfg.get_stream(rs.stream.color) # Fetch stream profile for depth stream
print(video_profile.as_video_stream_profile().get_intrinsics())