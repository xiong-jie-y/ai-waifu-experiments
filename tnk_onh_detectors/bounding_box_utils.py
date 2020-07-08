import numpy as np

import pyrealsense2 as rs

def oriented_to_axis_aligned(bounding_box_points_2d):
    return dict(
        min=(bounding_box_points_2d[:, 0].min(),
        bounding_box_points_2d[:, 1].min()),
        max=(bounding_box_points_2d[:, 0].max(),
        bounding_box_points_2d[:, 1].max())
    )

def min_max_to_4_points(mmbd):
    return np.array([
        list(mmbd['min']),
        [mmbd['min'][0], mmbd['max'][1]],
        [mmbd['max'][0], mmbd['max'][1]],
        [mmbd['max'][0], mmbd['min'][1]],
        list(mmbd['min']),
    ])

def bounding_box_from_points(points):
    return dict(
        min=[points[:, 0].min(), points[:, 1].min()],
        max=[points[:, 0].max(), points[:, 1].max()]
    )

def get_2d_min_max_bouding_box(oriented_bounding_box, intrinsic):
    points = np.array(oriented_bounding_box.get_box_points())
    # points_image = np.dot(points, camera_mat_np)
    # points_image = np.array(
    #     [camera_mat_np.dot([-point[1], -point[2], point[0]])[:2] for point in points]
    # )
    # print(points)
    # points_image = np.array(
    #     [camera_mat_np.dot([-point[1], point[2], point[0]])[:2] for point in points]
    # )
    points_image = np.array([rs.rs2_project_point_to_pixel(
        intrinsic,
        [-point[1], -point[2], point[0]]) for point in points])
    # points_image = np.array([[p[0], 480 - p[1]] for p in points_image])

    return oriented_to_axis_aligned(points_image)