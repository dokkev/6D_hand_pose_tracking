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

    
    def findpostion(self, img, pcl,camera_params):
        fx = camera_params.fx  # Focal length in pixels (x-axis)
        fy = camera_params.fy  # Focal length in pixels (y-axis)
        cx = camera_params.cx  # X-coordinate of the principal point
        cy = camera_params.cy  # Y-coordinate of the principal point


        data = []
        if self.results.multi_hand_landmarks:
             for landmarks in self.results.multi_hand_landmarks:
                for id, landmark in enumerate(landmarks.landmark):
                    # Find the pixel coordinates of the wrist
                    if id == 0:
                        h, w, _ = img.shape
                        cx, cy = int(landmark.x * w), int(landmark.y * h)

                        # circle cx, cy
                        cv2.circle(img, (cx, cy), 10, (0, 0, 255), -1)

                        # Use ZED point cloud to estimate 3D position of wrist
                        try:
                            err, point_cloud_value = pcl.get_value(cx, cy)
                            wrist_position = [point_cloud_value[0], point_cloud_value[1], point_cloud_value[2]]
                        except:
                            continue

                        # Use wrist position as starting point to project 2D landmarks into 3D space
                        for id, lm in enumerate(landmarks.landmark):
                
                            # print(wrist_position[0], (lm.x - landmark.x), wrist_position[2])
                            x_3d = wrist_position[0] + (lm.x - landmark.x) * wrist_position[2]
                            
                            y_3d = wrist_position[1] + (lm.y - landmark.y) * wrist_position[2]
                            z_3d = wrist_position[2]
                            hand_landmarks_3d = [x_3d, y_3d, z_3d]
                
                            data.append(hand_landmarks_3d)
    
        # Convert the data to a numpy array
        data = np.array(data)
        if data.shape != (21,3):
            # print without newline
            sys.stdout.write("\rNot all landmarks detected")
            sys.stdout.flush()

        else:
            sys.stdout.write("\rAll 21 landmarks detected")
            sys.stdout.flush()

 

        return data

            
    
    def displayFPS(self, img):
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
            ax.set_xlim3d(-0.2, 0.2)
            ax.set_ylim3d(-0.2, 0.2)
            ax.set_zlim3d(0.2, 1.0)
            ax.scatter3D(*zip(*data))

     
            edges = [(1,2),(2,3),(3,4),(0,5),(5,6),(5,9),(1,0),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]

            for edge in edges:
                ax.plot3D(*zip(data[edge[0]], data[edge[1]]), color='red')
      
            # Draw the plot
            plt.draw()
            plt.pause(0.0001)