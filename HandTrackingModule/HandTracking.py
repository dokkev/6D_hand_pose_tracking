import cv2
import time
import mediapipe as mp
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class HandTracking():
    def __init__(self, maxHands=2, detectionCon=0.2, trackCon=0.9):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False,
                                              max_num_hands= maxHands,           
                                              min_detection_confidence=detectionCon,   
                                              min_tracking_confidence=trackCon) 
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.time1 = time.time()
        self.wrist = []

    def findHands(self, img):
        
        img.flags.writeable = False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # flip the image horizontally for a selfie-view display.
        # img = cv2.flip(img, 1)
        self.results = self.hands.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


        if self.results.multi_hand_landmarks:
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
        h, w, _ = img.shape
        left_data = []
        right_data = []
        if self.results.multi_hand_landmarks:
             for landmarks in self.results.multi_hand_landmarks:
                handedness = self.results.multi_handedness[self.results.multi_hand_landmarks.index(landmarks)].classification[0].index
                for id, landmark in enumerate(landmarks.landmark):
                    
                    # Find the pixel coordinates of the wrist
                    if id == 0:
                        wrist_landmark_coordinate = [landmark.x, landmark.y, landmark.z]
                        X, Y = int(landmark.x * w), int(landmark.y * h)

                        # circle cx, cy
                        cv2.circle(img, (X, Y), 10, (0, 0, 255), -1)
                        # Use ZED point cloud to estimate 3D position of wrist
                        try:
                            err, point_cloud_value = pcl.get_value(X, Y)
                            wrist_position = [point_cloud_value[0], point_cloud_value[1], point_cloud_value[2]]
                        except:
                            continue

                    x_3d = wrist_position[0] + (landmark.x*w - wrist_landmark_coordinate[0]*w - cx) * wrist_position[2] / fx
                    y_3d = wrist_position[1] + (landmark.y*h - wrist_landmark_coordinate[1]*w - cy) * wrist_position[2] / fy
                    z_3d = wrist_position[2] + (landmark.z - wrist_landmark_coordinate[2]) * wrist_position[2]
                    hand_landmarks_3d = [x_3d, y_3d, z_3d]
                    # append the 3D position of each 3D landmark
                    if handedness == 1:
                        left_data.append(hand_landmarks_3d)
                        # put text left hand
                        cv2.putText(img, "Left", (X, Y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    elif handedness == 0:
                        right_data.append(hand_landmarks_3d) 
                        # put text right hand
                        cv2.putText(img, "Right", (X, Y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
               
   
        # Convert the data to a numpy array
        left_data = np.array(left_data)
        right_data = np.array(right_data)
        self.stdout_hand_detection(left_data,right_data)
      

        return left_data, right_data
    
    def calculate_orientation(self,hand_landmarks_3d):
        if hand_landmarks_3d.shape != (21,3):
            zero_array = np.zeros((3,))
            return  zero_array
        
        # Get the 3D positions of landmarks 0, 5, and 17
        wrist = hand_landmarks_3d[0]
        index = hand_landmarks_3d[5]
        pinky = hand_landmarks_3d[17]

        # Compute the vectors between the landmarks
        v1 = np.subtract(index, wrist)
        v2 = np.subtract(pinky, wrist)

        # Compute the normal vector to the plane defined by the landmarks
        normal = np.cross(v1, v2)
        normal /= np.linalg.norm(normal)

        # Compute the yaw, pitch, and roll angles based on the orientation of the normal vector
        yaw = np.arctan2(normal[1], normal[0])
        pitch = np.arctan2(-normal[2], np.sqrt(normal[0]**2 + normal[1]**2))
        roll = np.arctan2(np.sin(yaw)*v2[0]-np.cos(yaw)*v2[1], np.cos(yaw)*v1[1]-np.sin(yaw)*v1[0])

        self.orientation = np.array([yaw, pitch, roll])

        # Convert angles to degrees and return
        
        return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

    def calculate_centroid(self,hand_landmarks_3d):
        if hand_landmarks_3d.shape != (21,3):
            zero_array = np.zeros((3,))
            return  zero_array
        # Get the 3D positions of landmarks 0, 5, and 17
        
        wrist = hand_landmarks_3d[0]
        index = hand_landmarks_3d[5]
        pinky = hand_landmarks_3d[17]

        # Compute a middle point of three landmarks
        centroid = (wrist + index + pinky)/3



        return centroid

    def findNormalizedPosition(self,img):
        left_data = []
        right_data = []
        w, h, _ = img.shape

        if self.results.multi_hand_landmarks:
            
            for landmarks in self.results.multi_hand_landmarks:
                handedness = self.results.multi_handedness[self.results.multi_hand_landmarks.index(landmarks)].classification[0].index
                for id, landmark in enumerate(landmarks.landmark):
                    # Find the pixel coordinates of the wrist
                    if id == 0:
                        X, Y = int(landmark.x * h), int(landmark.y * w)
                        # circle X, Y
                        cv2.circle(img, (X, Y), 10, (0, 0, 255), -1)
         
                    hand_landmarks_3d = [landmark.x, landmark.y, landmark.z]
                    # append the 3D position of each 3D landmark 
                    if handedness == 1:
                        left_data.append(hand_landmarks_3d)
                        cv2.putText(img, "Left", (X, Y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    elif handedness == 0:
                        right_data.append(hand_landmarks_3d)
                        # put text left hand
                        cv2.putText(img, "Right", (X, Y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        left_data = np.array(left_data)
        right_data = np.array(right_data)
        self.stdout_hand_detection(left_data, right_data)

        return left_data, right_data
    
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
    

    def stdout_hand_detection(self, left_data, right_data):
        if left_data.shape == (21,3) and right_data.shape == (21,3):
            sys.stdout.write("\rLeft and Right hands all 21 landmarks detected")
            sys.stdout.flush()
        
        elif left_data.shape == (21,3) and right_data.shape != (21,3):
            sys.stdout.write("\rLeft hand all 21 landmarks detected")
            sys.stdout.flush()
        
        elif left_data.shape != (21,3) and right_data.shape == (21,3):
            sys.stdout.write("\rRight hand all 21 landmarks detected")
            sys.stdout.flush()
            
        else:
            sys.stdout.write("\rNo hand landmarks detected")
            sys.stdout.flush()
    

    def plot(self,ax,plt,data,xlim=(-0.5, 0.1),ylim=(-0.5, 0.1),zlim=(0.2, 1.0)):
        # Create 3D plot

        if data.shape >= (21,3):
          
            # Clear the plot and add new data
            ax.clear()
            
            # auto scale the plot
            # ax.autoscale(enable=True, axis='both', tight=None)

            ax.set_xlim3d(xlim)
            ax.set_ylim3d(ylim)
            ax.set_zlim3d(zlim)
            ax.scatter3D(*zip(*data))
     
            #  C
            edges = [(1,2),(2,3),(3,4),(0,5),(5,6),(5,9),(1,0),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
            edges2 = [(22,23),(23,24),(24,25),(21,26),(26,27),(26,30),(22,21),(27,28),(28,29),(21,30),(30,31),(31,32),(32,33),(30,34),(34,35),(35,36),(36,37),(34,38),(38,39),(39,40),(40,41),(21,38)] 

            if data.shape != (42,3):
                for edge in edges:
                    ax.plot3D(*zip(data[edge[0]], data[edge[1]]), color='red')

            else:
                for edge in edges:
                    ax.plot3D(*zip(data[edge[0]], data[edge[1]]), color='red')
                for edge in edges2:
                    ax.plot3D(*zip(data[edge[0]], data[edge[1]]), color='blue')


            # Draw the plot
            plt.draw()
            plt.pause(0.0001)