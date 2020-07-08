import pyrealsense2 as rs
import os
import open3d
from scipy.spatial.transform import Rotation
import numpy as np
import glob
import functools
import yaml
import re
from typing import *

# Wamt to hide this.
from matplotlib import pyplot as plt

import bounding_box_utils as bd_utils


def get_num(ext, s):
    m = re.search(r"(\d+)\." + ext, s)
    return int(m.group(1))


def show_rgbd_image(rgbd_image, points_image):
    plt.subplot(1, 2, 1)
    plt.title('SUN grayscale image')
    plt.imshow(rgbd_image.color)
    bd_2d = bd_utils.min_max_to_4_points(points_image)
    plt.plot(bd_2d[:, 0], bd_2d[:, 1])
    plt.subplot(1, 2, 2)
    plt.title('SUN depth image')
    plt.plot(bd_2d[:, 0], bd_2d[:, 1])
    plt.imshow(rgbd_image.depth)
    plt.show()
    print("shown")


def get_intrinsic():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)
    # pipeline.stop()
    cfg = pipeline.start(config)
    # Fetch stream profile for depth stream
    video_profile = cfg.get_stream(rs.stream.color)
    intri = video_profile.as_video_stream_profile().get_intrinsics()
    pipeline.stop()
    return intri


def bounding_box_2d_via_points(map_data, calib):
    bounding_box_2d_lists = []

    for pose, orientation in zip(map_data.positions, map_data.rotations):
        rot_inv = orientation.inv().as_matrix()
        annotation_list = []
        for label, point_cloud in map_data.annotations:
            points_cam = [
                rot_inv.dot(p)
                for p in (np.asfarray(point_cloud.points) - pose)]
            points_2d = np.array([rs.rs2_project_point_to_pixel(calib,
                                                                [-point[1], -point[2], point[0]]) for point in points_cam])
            annotation_list.append(
                (label, bd_utils.bounding_box_from_points(points_2d)))
        bounding_box_2d_lists.append(annotation_list)

    return bounding_box_2d_lists


def bounding_box_2d_via_3d_box(map_data, calib):
    bounding_box_lists = map_data.get_camera_centered_bounding_boxes()
    bounding_box_2d_lists = []

    for bounding_box_list in bounding_box_lists:
        bounding_box_2d_lists.append([
            (label, bd_utils.get_2d_min_max_bouding_box(bounding_box, calib))
            for label, bounding_box in bounding_box_list
        ])

    return bounding_box_2d_lists


class SLAMMap:
    def __init__(
            self, point_cloud: open3d.geometry.PointCloud,
            positions: np.array, rotations: np.array,
            rgbd_images: open3d.geometry.RGBDImage,
            calibration_info: dict, annotations: List[Tuple[Any]]):
        self.point_cloud = point_cloud
        self.positions = positions
        self.rotations = rotations
        self.rgbd_images = rgbd_images
        self.calibration_info = calibration_info
        self.annotations = annotations

    def get_2d_bounding_box_lists(self, bounding_box_func) -> List[List[Tuple[str, dict]]]:
        calib = get_intrinsic()
        return bounding_box_func(self, calib)

    def get_camera_centered_bounding_boxes(self) -> List[List[Tuple[str, open3d.geometry.OrientedBoundingBox]]]:
        camera_matrix = self.calibration_info['camera_matrix']
        camera_mat_np = np.array(camera_matrix['data']).reshape(
            camera_matrix['rows'], camera_matrix['rows'])
        print(camera_mat_np)

        mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.6, origin=[0, 0, 0])

        # Show Mesh Frame for a debug.
        mesh_frames = []
        for pos, rot in zip(self.positions, self.rotations):
            mesh_frame2 = open3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.2, origin=[0, 0, 0])
            mesh_frame2.rotate(rot.as_matrix(), np.array(
                [0, 0, 0])).translate(pos)
            mesh_frames.append(mesh_frame2)

        # o3d.visualization.draw_geometries(mesh_frames + [pcd2.get_oriented_bounding_box(), pcd, get_polyline_lineset(positions), mesh_frame])

        bd_cams = []
        for position, rotation in zip(self.positions, self.rotations):
            bounding_box_list = []
            for label, pcd2 in self.annotations:
                bd_cam = pcd2.get_oriented_bounding_box()
                bd_cam.translate(-position)
                print("before rot")
                print(np.array(bd_cam.R))
                # print(bd_cam.extent)
                # bd_cam.rotate(rotation.inv().as_matrix(), np.array([0,0,0]))
                bd_cam.R = rotation.inv().as_matrix().dot(bd_cam.R)
                # print(np.array(bd_cam.R))
                new_center = rotation.inv().as_matrix().dot(bd_cam.center)
                bd_cam.center = new_center
                print("after rot")
                # print(bd_cam.extent)
                # print(rotation.inv().as_matrix())
                print(np.array(bd_cam.R))
                # bd_cam = bd_cam
                bounding_box_list.append((label, bd_cam))
            bd_cams.append(bounding_box_list)

        return bd_cams

    def show_2d_dataset(self, bd_cam_tuples):
        intrinsic = get_intrinsic()
        min_max_bounding_boxes = []
        for i, bounding_box_list in enumerate(bd_cam_tuples):
            (label, bd_cam) = bounding_box_list[0]
            points_image = bd_utils.get_2d_min_max_bouding_box(
                bd_cam, intrinsic)
            # print(points_image)
            show_rgbd_image(self.rgbd_images[i], points_image)
            # min_max_bounding_boxes.append(points_)

    @classmethod
    def load_map_from_rtab_result_dir(cls, root_dir: str):
        pcd = open3d.io.read_point_cloud(os.path.join(root_dir, "cloud.ply"))

        positions = []
        rotations = []
        with open(os.path.join(root_dir, "poses.txt")) as f:
            for line in f.readlines():
                data = line.split(" ")
                position = [float(elm) for elm in data[1:4]]
                positions.append(np.array(position))
                rotations.append(Rotation.from_quat(
                    [float(elm) for elm in data[4:]]))

        # print(get_num("/home/yusuke/Documents/RTAB-Map/images/depth/1.png"))

        # Load RGBD images.
        depth_images = []
        depth_paths = list(
            glob.glob(os.path.join(root_dir, "images/depth/*.png")))
        depth_paths = sorted(
            depth_paths, key=functools.partial(get_num, "png"))
        for path in depth_paths:
            # print(path)
            depth_images.append(open3d.io.read_image(path))
        rgb_images = []
        rgb_paths = list(glob.glob(os.path.join(root_dir, "images/rgb/*.jpg")))
        rgb_paths = sorted(rgb_paths, key=functools.partial(get_num, "jpg"))
        for path in rgb_paths:
            # print(path)
            rgb_images.append(open3d.io.read_image(path))
        rgbd_images = []
        for depth_image, rgb_image in zip(depth_images, rgb_images):
            rgbd_images.append(open3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb_image, depth_image, convert_rgb_to_intensity=False))

        calib = yaml.load(
            open(os.path.join(root_dir, "images/calibration.yaml")))

        # TODO: Handle multple labels.
        pcd_path = os.path.join(root_dir, "annotated_data.ply")
        if os.path.exists(pcd_path):
            pcd2 = open3d.io.read_point_cloud(pcd_path)
            annotations = [('onahole', pcd2)]
        else:
            annotations = []

        return SLAMMap(pcd, positions, rotations, rgbd_images, calib, annotations)
