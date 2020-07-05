import numpy as np
import open3d

from PIL import Image
import io

# TODO: write test. How to handle different compression with png?
# import utils

# open("test.png", "wb").write(utils.convert_from_open3d_color_image_to_png(rgbd_images[0].color))
def convert_from_open3d_color_image_to_png(color_image: open3d.geometry.Image):
    roiImg = Image.fromarray(np.uint8(color_image))
    imgByteArr = io.BytesIO()
    roiImg.save(imgByteArr, format='PNG')
    imgByteArr = imgByteArr.getvalue()

    return imgByteArr, roiImg.size[0], roiImg.size[1]