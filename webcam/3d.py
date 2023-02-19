import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from cvzone.HandTrackingModule import HandDetector


# Start webcam
cap = cv2.VideoCapture(0)
width, height = 1280, 720
cap.set(3, width)
cap.set(4, height)

# Create hand detector
detector = HandDetector(detectionCon=0.8)

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

while True:
    # Get image frame
    success, img = cap.read()

    # Find the hands
    hands, img = detector.findHands(img)

    # Landmark values - (x,y,z)*21
    data = []
    if hands:
        # First hand
        hand = hands[0]
        lmList = hand["lmList"] # List of 21 Landmark points
        bbox = hand["bbox"] # Bounding box info x,y,w,h
        centerPoint = hand['center'] # center of the hand cx,cy
        handType = hand["type"] # Handtype Left or Right
        

        for lm in lmList:
            data.append([lm[0], lm[1], lm[2]])


        print(data)



        # Clear the plot and add new data
        ax.clear()
        ax.set_xlim3d(0, width)
        ax.set_ylim3d(0, height)
        ax.set_zlim3d(-200, 200)
        ax.scatter3D(*zip(*data))

        # Draw the plot
        plt.draw()
        plt.pause(0.0000000000000000000000000000000000001)

    # Display the image frame
    cv2.imshow("Image", img)
    cv2.waitKey(1)