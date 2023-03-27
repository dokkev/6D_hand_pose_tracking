import cv2
import sys,os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import mediapipe as mp
import numpy as np
from HandTrackingModule.HandTracking import HandTracking
from HandTrackingModule.Zed import Zed
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
    Lx, Ly, Lz, Lyaw, Lpitch, Lroll = 0, 0, 0, 0, 0, 0
    Rx, Ry, Rz, Ryaw, Rpitch, Rroll = 0, 0, 0, 0, 0, 0


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
            data_left,data_right = detector.findpostion(depth_img, pcl,camera_params)

            # find the orientation of the palm
            left_orientaton = detector.calculate_orientation(data_left)
            # find the centroid of the palm
            left_centroid = detector.calculate_centroid(data_left)

            # find the orientation of the palm
            right_orientaton = detector.calculate_orientation(data_right)
            # find the centroid of the palm
            right_centroid = detector.calculate_centroid(data_right)

            if len(data_left) != 0:
                lx = left_centroid[0]
                ly = left_centroid[1]
                lz = left_centroid[2]
                lyaw = left_orientaton[0]
                lpitch = left_orientaton[1]
                lroll = left_orientaton[2]

            if len(data_right) != 0:
                rx = right_centroid[0]
                ry = right_centroid[1]
                rz = right_centroid[2]
                ryaw = right_orientaton[0]
                rpitch = right_orientaton[1]
                rroll = right_orientaton[2]

            F.append(frame)
            LX.append(lx)
            LY.append(ly)
            LZ.append(lz)
            LYAW.append(lyaw)
            LPITCH.append(lpitch)
            LROLL.append(lroll)
            RX.append(rx)
            RY.append(ry)
            RZ.append(rz)
            RYAW.append(ryaw)
            RPITCH.append(rpitch)
            RROLL.append(rroll)

            df = pd.DataFrame(name_dict)
            df.to_csv('results/demo.csv',index=False)

            # system out frame number without newline
            print(" | Frame count: ",frame, "/",final_frame, end='\r')




if __name__ == "__main__":
    #  Initialize lists for Pandas DataFrame
    F, RX, RY, RZ, RYAW, RPITCH, RROLL = [], [], [], [], [], [], []
    LX, LY, LZ, LYAW, LPITCH, LROLL = [], [], [], [], [], [], []
    key = ' '
    name_dict = {
        'Frame': F,
        'left X': RX,
        'left Y': RY,
        'left Z': RZ,
        'left Yaw': RYAW,
        'left Pitch': RPITCH,
        'left Roll': RROLL,
        'right X': LX,
        'right Y': LY,
        'right Z': LZ,
        'right Yaw': LYAW,
        'right Pitch': LPITCH,
        'right Roll': LROLL,     

    }

    # run main
    main()