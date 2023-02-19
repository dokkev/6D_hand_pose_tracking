import cv2
import mediapipe as mp
import time
import pyzed.sl as sl
import sys
import math
import numpy as np
import csv
import pandas as pd

# targe pixel coordinates
global XY

pTime = 0

def euclidean_distance(X,Y,Z):
    """
    return euclidean distance from a 3d point
    arg: [x y z]
    return: distance 
    """
    distance = math.sqrt(X**2 + Y**2 + Z**2)
    return distance
                 

# Create a ZED camera object
zed = sl.Camera()

# Set configuration parameters
# camera
input_type = sl.InputType()
if len(sys.argv) >= 2 :
    input_type.set_from_svo_file(sys.argv[1])

use_svo = True

# svo
if use_svo == True:
    input_type = sl.InputType()
    input_type.set_from_svo_file("12-8-collectoin.svo")

init_params = sl.InitParameters(input_t=input_type)
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.depth_mode = sl.DEPTH_MODE.QUALITY
init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
init_params.depth_minimum_distance = 0.015


# Open the camera
err = zed.open(init_params)
if err != sl.ERROR_CODE.SUCCESS :
    print(repr(err))
    zed.close()
    exit(1)


# Create and set RuntimeParameters after opening the camera
runtime_parameters = sl.RuntimeParameters()
runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL  # Use FILL sensing mode
# runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  # Use STANDARD sensing mode
# Setting the depth confidence parameters
runtime_parameters.confidence_threshold = 100
runtime_parameters.textureness_confidence_threshold = 100


# declare image and depth
image = sl.Mat()
depth = sl.Mat()
point_cloud = sl.Mat()
sensors_data = sl.SensorsData()
ts = sl.Timestamp()

mirror_ref = sl.Transform()
mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
tr_np = mirror_ref.m

# image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
# depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
point_cloud = sl.Mat()

#  Hand Tracking Module
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
                      min_detection_confidence=0.2,
                      min_tracking_confidence=0.8)
mpDraw = mp.solutions.drawing_utils

x = 0
y = 0
z = 0
X = []
Y = []
Z = []
F = []



frame = 0
end = False
key = ' '
name_dict = {
    'Frame': F,
    'X': X,
    'Y': Y,
    'Z': Z,

}

print("Resolution: {0}, {1}.".format(round(zed.get_camera_information().camera_resolution.width, 2), zed.get_camera_information().camera_resolution.height))
print("Camera FPS: {0}".format(zed.get_camera_information().camera_fps))
print("Frame count: {0}.\n".format(zed.get_svo_number_of_frames()))

while end == False :
    err = zed.grab(runtime_parameters)
    if err == sl.ERROR_CODE.SUCCESS :
        # Retrieve the left color image
        zed.retrieve_image(image, sl.VIEW.LEFT)
        # Retrieve the aligned depth to left color image
        zed.retrieve_image(depth, sl.VIEW.DEPTH)
        # Retrieve the RGBA point cloud in half resolution
        zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)

        # keep track of frames
        frame += 1
        final_frame = zed.get_svo_number_of_frames()
   

        # convert image to numpy array
        img = image.get_data()
        depth_image_ocv = depth.get_data()


        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)
        
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        cx_min = 700
        cx_max = 1200

        img = cv2.line(img,(cx_min,900),(cx_min,0),(0, 255, 0),8)
        img = cv2.line(img,(cx_max,900),(cx_max,0),(0, 255, 0),8)

    

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                
                for id, lm in enumerate(handLms.landmark):
                    
                   
                    h, w, c = img.shape
                    cx, cy = int(lm.x *w), int(lm.y*h)
                    if cx_min < cx < cx_max:
                        if id ==8:
                            XY = [cx,cy]
                            cv2.circle(img, (cx,cy), 30, (255,255,255), cv2.FILLED)
                            cv2.circle(depth_image_ocv, (cx,cy), 30, (255,0,255), cv2.FILLED)

                            try: 
                                err,point_cloud_value = point_cloud.get_value(cx, cy)
                                point_cloud_np = point_cloud.get_data()
                                # point_cloud_np.dot(tr_np)

                                # print("Wrist Coordinate: ({:1.3}, {:1.3},{:1.3})".format(point_cloud_value[0], point_cloud_value[1],point_cloud_value[2], end="\r"))
                                
                                x = point_cloud_value[0]
                                y = point_cloud_value[1]
                                z = point_cloud_value[2]
                        

                            except Exception:
                                pass

            
            
            pass

        F.append(frame)
        X.append(x)
        Y.append(y)
        Z.append(z)
        df = pd.DataFrame(name_dict)
        df.to_csv('normal_person_right.csv',index=False)
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imshow("Image", img)
        # cv2.imshow("Depth", depth_image_ocv)
        key = cv2.waitKey(1)
        if frame == final_frame:
            end=True

    else:
        key = cv2.waitKey(1)


print("saved")
cv2.destroyAllWindows()
zed.close()

