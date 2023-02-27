import cv2
import time
import mediapipe as mp
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


    
class HandTracking():
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                              max_num_hands=1,           
                                              min_detection_confidence=0.5,   
                                              min_tracking_confidence=0.5) 
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.time1 = time.time()
        self.wrist = []

    def findHands(self, img):
      
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # flip the image horizontally for a selfie-view display.
        # img = cv2.flip(img, 1)
        self.results = self.hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)



        if self.results.multi_hand_landmarks:
            # print(self.results.multi_handedness)
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(), 
                    self.mp_styles.get_default_hand_connections_style()
                )        
        return img

    def findPosition(self):
        data = []
       
        # if self.results.multi_hand_landmarks:
            # print(self.results.multi_hand_landmarks)
            # for hand_landmarks in self.results.multi_hand_landmarks:
            #     for id, lm in enumerate(hand_landmarks.landmark):
   
            # # for hand_world_landmarks in self.results.multi_hand_world_landmarks:
            # #     for id, lm in enumerate(hand_world_landmarks.landmark):
            # #         data.append([lm.x, lm.y, lm.z])

        
        # return data
    
    def find3Dpostion(self, depth_img, pcl,camera_params):
        fx = camera_params.fx  # Focal length in pixels (x-axis)
        fy = camera_params.fy  # Focal length in pixels (y-axis)
        cx = camera_params.cx  # X-coordinate of the principal point
        cy = camera_params.cy  # Y-coordinate of the principal point


        data = []
        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
                    if id == 0:
                        # Find the pixel coordinates of the wrist
                        h, w, c = depth_img.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        try:
                            err, point_cloud_value = pcl.get_value(cx, cy)
                            x = point_cloud_value[0]
                            y = point_cloud_value[1]
                            z = point_cloud_value[2]
                            self.wrist = [x, y, z]
                            # print("Wrist: ", self.wrist)
                            # Mark on depth image
                            cv2.circle(depth_img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

                          
                            print("cx: ",)
                            print("x: ",lm)
                        except Exception:
                            print("Error: ", Exception)
                            pass
      

        # Convert the data to a numpy array
        data = np.array(data)





            
                    
    
    def displayFPS(self, cap, img):
        # Set the time for this frame to the current time.
        self.time2 = time.time()
        # Check if the difference between the previous and this frame time > 0 to avoid division by zero.
        if (self.time2 - self.time1) > 0:
        
            # Calculate the number of frames per second.
            frames_per_second = 1.0 / (self.time2 - self.time1)
            
            # Write the calculated number of frames per second on the frame. 
            cv2.putText(img, 'FPS: {}'.format(int(frames_per_second)), (10, 30),cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)
            self.time1 = self.time2
        
        return img
    

    def plot(self,ax,plt,data):
        # Create 3D plot

        if self.results.multi_hand_landmarks:
            # Clear the plot and add new data
            ax.clear()
            ax.set_xlim3d(-0.5, 0.5)
            ax.set_ylim3d(-0.5, 0.5)
            ax.set_zlim3d(-0.5, 0.5)
            ax.scatter3D(*zip(*data))

     
            edges = [(1,2),(2,3),(3,4),(0,5),(5,6),(5,9),(1,0),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]

            # for edge in edges:
                # ax.plot3D(*zip(data[edge[0]], data[edge[1]]), color='red')
      
            # Draw the plot
            plt.draw()
            plt.pause(0.0001)