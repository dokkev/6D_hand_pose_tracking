import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands


# Initialize the parameters for the hands tracking
hands = mp_hands.Hands(
    static_image_mode=False,        # If set to false, the solution treats the input images as a video stream. 
                                    # It will try to detect hands in the first input images, and upon a successful detection further localizes the hand landmarks. 
                                    # In subsequent images, once all max_num_hands hands are detected and the corresponding hand landmarks are localized, 
                                    # it simply tracks those landmarks without invoking another detection until it loses track of any of the hands. 
                                    # This reduces latency and is ideal for processing video frames. If set to true, hand detection runs on every input image, 
                                    # ideal for processing a batch of static, possibly unrelated, images.

    max_num_hands=1,                # Maximum number of hands to detect

    min_detection_confidence=0.5,   # Minimum confidence value ([0.0, 1.0]) from the hand detection model for the detection to be considered successful.

    min_tracking_confidence=0.5     # Minimum confidence value ([0.0, 1.0]) from the landmark-tracking model for the hand landmarks to be considered tracked successfully, 
                                    # or otherwise hand detection will be invoked automatically on the next input image. 
                                    # Setting it to a higher value can increase robustness of the solution, at the expense of a higher latency. 
    )                               # Ignored if static_image_mode is true, where hand detection simply runs on every image. 

#  Define the camera with cv2
cap = cv2.VideoCapture(0)


while cap.isOpened():
    success, image = cap.read()
    # Exit the loop if no image is captured
    if not success:
        print("Error: No Camera Found")
        break

    # Convert the image to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Process the image
    results = hands.process(image)
    # convert the image back to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Collection of detected/tracked hands, where each hand is represented as a list of 21 hand landmarks
    if results.multi_hand_landmarks:
        # for-loop for each landmark of each hand
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the landmarks
            mp_drawing.draw_landmarks(
                image, # define the image
                hand_landmarks, # display dots on the landmarks
                mp_hands.HAND_CONNECTIONS, # connect the dots between the landmarks
                mp_drawing_styles.get_default_hand_landmarks_style(), # change the color of dots for different regions of the hand
                mp_drawing_styles.get_default_hand_connections_style() # change the color of the lines between the dots
            )
            
    
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    # Display FPS
    cv2.putText(image, f'FPS: {int(cap.get(cv2.CAP_PROP_FPS))}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Exit the loop if the user press the 'Esc' key
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()