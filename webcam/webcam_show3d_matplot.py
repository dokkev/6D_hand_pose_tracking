import sys,os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from HandTrackingModule.HandTracking import HandTracking as ht
import cv2
import time
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

def main():
    cap = cv2.VideoCapture(0)
    print("Live Streaming from webcam")
    cap.set(3,1280)
    cap.set(4,960)

    detector = ht()
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

        img= detector.findHands(img)
        data_left, data_right = detector.findNormalizedPosition(img)
 

        

        if plot == True:
            xlim = (-0.1, 0.8)
            ylim = (-0.1, 0.8)
            zlim = (0, 0.5)
            if data_right.shape == (21,3) and data_left.shape == (21,3):
                data_plot = np.vstack((data_right,data_left))
                print()
            elif data_right.shape == (21,3):
                data_plot = data_right
            elif data_left.shape == (21,3):
                data_plot = data_left
            else:
                data_plot = np.array([])
            detector.plot(ax,plt,data_plot, xlim, ylim, zlim)
         



        img = detector.displayFPS(img)
        cv2.imshow("MediaPipe Hands", img)
        cv2.waitKey(1)

if __name__ == "__main__":

    main()
    


