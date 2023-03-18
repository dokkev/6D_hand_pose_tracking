import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import mediapipe as mp
import cv2
import pyzed.sl as sl
import numpy as np
import matplotlib.pyplot as plt
from HandTrackingModule.HandTracking import HandTracking
from HandTrackingModule.Zed import Zed


def main():

    if len(sys.argv) == 2:
        print("Camera Mode: SVO")
        filename = sys.argv[1]
    else:
        print("Camera Mode: Live Streaming")
        filename = None
        
    # bring detector
    detector = HandTracking()
    # bring zed
    cam = Zed(filename)
    
    # print camera information
    cam.print_information()
    camera_params = cam.camera_params

    # 3d plot
    # Create 3D plot
    plot = True
    if plot == True:
        fig = plt.figure()
        plt.ion()
        ax = fig.add_subplot(111, projection='3d')

    while True:
        err = cam.zed.grab(cam.runtime_parameters)
        if err == sl.ERROR_CODE.SUCCESS:
   
            # extract images from ZED
            cam.get_image()
            img = cam.img
            depth_img = cam.depth_img

            # Point cloud data
            pcl = cam.point_cloud

            # find hands
            img = detector.findHands(img)
            # find position
            data = detector.findpostion(depth_img, pcl,camera_params)
   

            # half size images
            img = cv2.resize(img, (0, 0), None, 0.5, 0.5)
            depth_img = cv2.resize(depth_img, (0, 0), None, 0.5, 0.5)


            if plot == True:
                detector.plot(ax,plt,data)

            detector.displayFPS(img)
            cv2.imshow("Image", img)
            cv2.imshow("Depth", depth_img)
     
            cv2.waitKey(1)

 
if __name__ == "__main__":
    main()