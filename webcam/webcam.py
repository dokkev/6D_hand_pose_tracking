import cv2
import mediapipe as mp
import sys

# Load the MediaPipe hand tracking model
mp_hands = mp.solutions.hands

# Initialize the hand tracking module
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open the video stream
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    success, image = cap.read()
    if not success:
        break

    # Convert the image to RGB format
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Run the hand tracking model on the image
    results = hands.process(image)

    # Extract the 6D hand pose from the results
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        pose = []
        for lm in hand_landmarks.landmark:
            x = lm.x
            y = lm.y
            z = lm.z

        # print x, y, z without printing a new line
        sys.stdout.write("\r%f %f %f" % (x, y, z))
        sys.stdout.flush()
        


       


       

    # Display the image with the hand pose
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    cv2.imshow("Hand Pose Tracking", image)

    # Exit on key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and the hand tracking module
cap.release()
hands.close()