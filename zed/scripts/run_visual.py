import cv2
import mediapipe as mp
import time
import pyzed.sl as sl
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Zed():
    def __init__(self):

        # Decide if you want to use the SVO file or the camera
        use_svo = False
        self.input_type = sl.InputType()
        if use_svo == False:
            print("Using SVO file")
            self.input_type = sl.InputType()
            self.input_type.set_from_svo_file("svo/dr_vernaldos_estamoses.svo")
       
        
        # Initialize the ZED camera
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters(input_t=self.input_type)
        self.init_params.camera_resolution = sl.RESOLUTION.HD720
        self.init_params.depth_mode = sl.DEPTH_MODE.QUALITY
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.depth_minimum_distance = 0.015

        # Open the camera
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS :
            print(repr(err))
            self.zed.close()
            exit(1)

        # Create and set RuntimeParameters after opening the camera
        self.runtime_parameters = sl.RuntimeParameters()
        self.runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL  # Use FILL sensing mode
        # Setting the depth confidence parameters
        self.runtime_parameters.confidence_threshold = 100
        self.runtime_parameters.textureness_confidence_threshold = 100


        # declare image, depth, nd point cloud
        self.image = sl.Mat()
        self.depth = sl.Mat()
        self.point_cloud = sl.Mat()

    def print_information(self):
        print("Resolution: {0}, {1}.".format(round(self.zed.get_camera_information().camera_resolution.width, 2), self.zed.get_camera_information().camera_resolution.height))
        print("Camera FPS: {0}".format(self.zed.get_camera_information().camera_fps))
        print("Frame count: {0}.\n".format(self.zed.get_svo_number_of_frames()))


    def get_image(self):
        # Retrieve left image
        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
        # Retrieve depth map. Depth is aligned on the left image
        self.zed.retrieve_image(self.depth, sl.VIEW.DEPTH)
        # Retrieve colored point cloud. Point cloud is aligned on the left image.
        self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)

        # convert zed image to numpy array
        self.img = self.image.get_data()
        self.depth_img = self.depth.get_data()


    


class handDetector():
    def __init__(self):

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
        self.mpDraw = mp.solutions.drawing_utils

        self.point = [0,0,0]
        
    def findHands(self,img, draw = True):
        imgBGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.results = self.hands.process(imgBGR)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(imgBGR, handLms, self.mpHands.HAND_CONNECTIONS)

        
        img = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
        return img

    def findPosition(self, img,depth_img, pcl, handNo = 0, draw = True):

            lmlist = []
            # draw lines


            if self.results.multi_hand_landmarks:
                for handLms in self.results.multi_hand_landmarks:
                    for id, lm in enumerate(handLms.landmark):
                        # print(lm)

                        h, w, c = img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                
                        for handLms in self.results.multi_hand_landmarks:
                            self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
                            if draw:
                                cv2.circle(depth_img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)

                        lmlist.append([id, cx, cy])

                        # pointer finger pcl
                        if id == 8:
                            try: 
                                err,point_cloud_value = pcl.get_value(cx, cy)



                                cv2.circle(depth_img, (cx, cy), 10, (0, 255, 255), cv2.FILLED)
                                cv2.circle(img, (cx, cy), 10, (0, 255, 255), cv2.FILLED)
                            
                        

                                print("Wrist Coordinate: ({:1.3}, {:1.3},{:1.3})".format(point_cloud_value[0], point_cloud_value[1],point_cloud_value[2], end="\r"))
                                
                                x = point_cloud_value[0]
                                y = point_cloud_value[1]
                                z = point_cloud_value[2]



                                self.point = [x,y,z]

                        
                                                               
                    

                            except Exception:
                                pass
                            
            return lmlist
def main():
    # create list store data

    end = False

    # bring detector
    detector = handDetector()
    # bring zed
    cam = Zed()


    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.show()
  

    while end == False:
        err = cam.zed.grab(cam.runtime_parameters)
        if err == sl.ERROR_CODE.SUCCESS:
            # keep track of frame
            # frame += 1

            # extract images from ZED
            cam.get_image()
            img = cam.img
            depth_img = cam.depth_img

            # Point cloud data
            pcl = cam.point_cloud

            # find hands
            img = detector.findHands(img)
            lmlist = detector.findPosition(img,depth_img,pcl)
            
            px = int(720*1.3)
            py = int(360*1.3)
            img = cv2.resize(img, (px, py))
            depth_img = cv2.resize(depth_img, (px, py))
  
            cv2.imshow("Image", img)
            cv2.imshow("Depth", depth_img)
    
    
            cv2.waitKey(1)


    

 
if __name__ == "__main__":
    main()