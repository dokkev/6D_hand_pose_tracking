import cv2
import numpy as np
import mediapipe as mp
import sys

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Camera intrinsics and distortion coefficients
# These should be obtained by calibrating your camera
K = np.array([[630.5222, 0, 325.6167],
              [0, 631.4028, 240.8563],
              [0, 0, 1]], dtype=np.float64)
dist = np.array([-0.2078, 0.0943, -0.0003, -0.0012, 0], dtype=np.float64)

# Distance between camera and hand in meters
distance = 0.5

with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Undistort the input image
        h, w = image.shape[:2]
        newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))
        mapx, mapy = cv2.initUndistortRectifyMap(K, dist, None, newcameramatrix, (w, h), 5)
        image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Extract the 3D hand pose in meters from the results
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand_landmarks.landmark:
                x = lm.x * w
                y = lm.y * h
                z = lm.z
                # Convert normalized coordinates to meters
                X, Y, Z = np.dot(np.linalg.inv(K), np.array([x, y, 1])) * z * distance
                landmarks.append([X, Y, Z])

            # print X, Y, Z without printing a new line
            sys.stdout.write("\r%f %f %f" % (X, Y, Z))
            sys.stdout.flush()


        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('Hand Tracking', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break
    cap.release()