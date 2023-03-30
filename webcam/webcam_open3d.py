import sys,os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from HandTrackingModule.HandTracking import HandTracking as ht
import cv2
import numpy as np
import open3d as o3d


def main():
    cap = cv2.VideoCapture(0)
    print("Live Streaming from webcam")


    detector = ht()

    # Create Open3D visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry(mesh_frame)

    # Initialize left and right hand point clouds
    pcd_left = o3d.geometry.PointCloud()
    pcd_left.paint_uniform_color([1, 0.706, 0])
    vis.add_geometry(pcd_left)

    pcd_right = o3d.geometry.PointCloud()
    pcd_right.paint_uniform_color([0, 0.651, 0.929])
    vis.add_geometry(pcd_right)


    while cap.isOpened():
        success, img = cap.read()

        if not success:
            print("Error: No Camera Found")
            break

        img = detector.findHands(img)
        data_left, data_right = detector.findNormalizedPosition(img)

        # Create Open3D point clouds for the left and right hands if they exist
        if data_right.shape == (21, 3):
            vis.remove_geometry(pcd_right)
            pcd_right = o3d.geometry.PointCloud()
            pcd_right.points = o3d.utility.Vector3dVector(data_right)
            pcd_right.paint_uniform_color([1, 0, 0])
            vis.add_geometry(pcd_right)

        if data_left.shape == (21, 3):
            vis.remove_geometry(pcd_left)
            pcd_left = o3d.geometry.PointCloud()
            pcd_left.points = o3d.utility.Vector3dVector(data_left)
            pcd_left.paint_uniform_color([0, 0, 1])
  
            vis.add_geometry(pcd_left)

  
  

            # Update visualization
            vis.poll_events()
            vis.update_renderer()

        img = detector.displayFPS(img)
        cv2.imshow("MediaPipe Hands", img)
        cv2.waitKey(1)

    # Close Open3D visualization window
    vis.destroy_window()


if __name__ == "__main__":
    main()
    


