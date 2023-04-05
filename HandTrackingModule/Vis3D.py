import open3d as o3d

"""
Class for visualizing the hand landmarks and camera using open 3D
"""

class Vis3D():
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        self.vis.add_geometry(self.mesh_frame)

        # Initialize left and right hand point clouds
        self.pcd_left = o3d.geometry.PointCloud()
        self.pcd_left.paint_uniform_color([1, 0.706, 0])
        self.vis.add_geometry(self.pcd_left)

        self.pcd_right = o3d.geometry.PointCloud()
        self.pcd_right.paint_uniform_color([0, 0.651, 0.929])
        self.vis.add_geometry(self.pcd_right)

        self.pcd_hand = o3d.geometry.PointCloud()
        self.pcd_hand.paint_uniform_color([0, 0.651, 0.929])
        self.vis.add_geometry(self.pcd_hand)

        self.blue = [0, 0, 1]
        self.red = [1, 0, 0]


    def show_hand(self,data,color=[0, 0, 1]):
        # return none if no data
        if data.shape != (21,3):
            return None
        else:
            self.vis.remove_geometry(self.pcd_hand)
            self.pcd_hand = o3d.geometry.PointCloud()
            self.pcd_hand.points = o3d.utility.Vector3dVector(data)
            self.pcd_hand.paint_uniform_color(color)
            self.vis.add_geometry(self.pcd_hand)

              # Update visualization
            self.vis.poll_events()
            self.vis.update_renderer()

