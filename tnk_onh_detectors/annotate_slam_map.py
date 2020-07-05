#%%
import slam_map
import os
import open3d
import numpy as np

map = slam_map.SLAMMap.load_map_from_rtab_result_dir(
    os.path.expanduser("~/Documents/RTAB-Map/200704-143117")
)
def annotate_point_cloud(pcd):
    vis = open3d.visualization.VisualizerWithEditing()
    vis.create_window()
    # vis.add_geometry(line_set, True)
    # vis.run()
    vis.add_geometry(pcd, True)
    vis.update_renderer()
    view_control = vis.get_view_control()
    view_control.set_lookat(np.array([0,0,0]))
    vis.run()

    # vis.add_geometry(line_set)

    vis.destroy_window()

annotate_point_cloud(map.point_cloud)

# %%
