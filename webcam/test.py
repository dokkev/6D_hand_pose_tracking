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

    def findHands(self, img):
      
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # flip the image horizontally for a selfie-view display.
        img = cv2.flip(img, 1)
        self.results = self.hands.process(img)
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

    def findPosition(self):
        data = []
        # print(self.results.multi_handedness)
        if self.results.multi_hand_landmarks:
            # print(self.results.multi_hand_landmarks)
            for hand_landmarks in self.results.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
                    data.append([lm.x, lm.y, lm.z])
            # for hand_world_landmarks in self.results.multi_hand_world_landmarks:
            #     for id, lm in enumerate(hand_world_landmarks.landmark):
            #         data.append([lm.x, lm.y, lm.z])

        data = np.array(data)
        return data
            
    

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,960)

    detector = HandTracking()
    plot = True

    # Create 3D plot
    if plot == True:
        fig = plt.figure()
        plt.ion()
        ax = fig.add_subplot(111, projection='3d')

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            print("Error: No Camera Found")
            break

        img = detector.findHands(img)
        data = detector.findPosition()

        if plot == True:
            detector.plot(ax,plt,data)


        img = detector.displayFPS(cap,img)
        cv2.imshow("MediaPipe Hands", img)
        cv2.waitKey(1)

if __name__ == "__main__":

   
    
    main()
    


