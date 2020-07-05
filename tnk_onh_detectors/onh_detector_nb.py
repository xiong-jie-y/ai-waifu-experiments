#%%
import numpy as np
import open3d as o3d
import os
import glob

def get_polyline_lineset(positions):
    lines = [[i, i+1] for i in range(0, len(positions) - 1)]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(positions),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set

from scipy.spatial.transform import Rotation




import re
import functools as ft

import yaml



o3d.visualization.draw_geometries(bd_cams + mesh_frames + [pcd2.get_oriented_bounding_box(), pcd, get_polyline_lineset(positions), mesh_frame])

#%%
print(positions[0])
print(rotations[0].inv().as_quat())
print(rotations[0].inv().apply([1,0,0]))
print(np.array(pcd2.get_oriented_bounding_box().get_box_points()))
np.array(bd_cams[0].get_box_points())

#%%
bd_cams[0].extent
bd_cams[0].R
bd_cams[0].center

#%%
camera_mat_np.dot([1,0,-10])

#%%


rs.rs2_project_point_to_pixel(intri, [1,0,10])

#%%



import matplotlib.pyplot as plt


# [np.array(bd_cam.get_box_points()) for bd_cam in bd_cams]


# o3d.visualization.draw_geometries([line_set, pcd, bd])

#%%
o3d.visualization.draw_geometries([rgb_images[0]])

# o3d.visualization.draw_geometries([], zoom=0.8)

# %%
import open3d

def key_action_callback(vis, action, mods):
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=10.0,
                                                height=10.0,
                                                depth=10.0)
    mesh_box.compute_vertex_normals()
    mesh_box.paint_uniform_color([0.9, 0.1, 0.1])

    print(action)
    if action == 1:  # key down
        vis.add_geometry(mesh_box)
    elif action == 0:  # key up
        pass
    elif action == 2:  # key repeat
        pass
    return True

vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window()
# vis.add_geometry(line_set, True)
# vis.run()
vis.add_geometry(pcd, True)
vis.update_renderer()
view_control = vis.get_view_control()
view_control.set_lookat(np.array([0,0,0]))
vis.register_key_action_callback(32, key_action_callback)
vis.run()

# vis.add_geometry(line_set)

vis.destroy_window()
print("")

# picked_points = vis.get_picked_points()
# print(picked_points)

# o3d.visualization.draw_geometries([line_set, pcd])


# %%
import matplotlib.pyplot as plt

def show_rgbd_image(rgbd_image):
    plt.subplot(1, 2, 1)
    plt.title('SUN grayscale image')
    plt.imshow(rgbd_image.color)
    plt.subplot(1, 2, 2)
    plt.title('SUN depth image')
    plt.imshow(rgbd_image.depth)
    plt.show()

show_rgbd_image(rgbd_images[0])
# o3d.visualization.draw_geometries([rgbd_images[0]], zoom=0.5)

# %%
