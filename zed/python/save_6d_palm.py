import cv2
import sys
import mediapipe as mp
import numpy as np
from Zed import Zed
from MpHand import HandTracking
import pyzed.sl as sl
import pandas as pd


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
    final_frame = cam.zed.get_svo_number_of_frames()
    camera_params = cam.camera_params
    frame = 0
    x, y, z, yaw, pitch, roll = 0, 0, 0, 0, 0, 0


    while frame <= final_frame:
        err = cam.zed.grab(cam.runtime_parameters)
        if err == sl.ERROR_CODE.SUCCESS:
            # increment frame
            frame += 1
            # extract images from ZED
            cam.get_image()
            img = cam.img
            depth_img = cam.depth_img

            # Point cloud data
            pcl = cam.point_cloud

            # find hands
            img = detector.findHands(img)
            # find handmark positions
            data = detector.findpostion(depth_img, pcl,camera_params)

            # find the orientation of the palm
            orientaton = detector.calculate_orientation(data)
            # find the centroid of the palm
            centroid = detector.calculate_centroid(data)

            if len(data) != 0:
                x = centroid[0]
                y = centroid[1]
                z = centroid[2]
                yaw = orientaton[0]
                pitch = orientaton[1]
                roll = orientaton[2]

            F.append(frame)
            X.append(x)
            Y.append(y)
            Z.append(z)
            YAW.append(yaw)
            PITCH.append(pitch)
            ROLL.append(roll)

            df = pd.DataFrame(name_dict)
            df.to_csv('../results/demo.csv',index=False)

            # system out frame number without newline
            print(" | Frame count: ",frame, "/",final_frame, end='\r')



   








 
if __name__ == "__main__":
    #  Initialize lists for Pandas DataFrame
    F, X, Y, Z, YAW, PITCH, ROLL = [], [], [], [], [], [], []
    key = ' '
    name_dict = {
        'Frame': F,
        'X': X,
        'Y': Y,
        'Z': Z,
        'Yaw': YAW,
        'Pitch': PITCH,
        'Roll': ROLL
    }


    # run main
    main()