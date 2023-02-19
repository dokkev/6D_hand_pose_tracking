import cv2
import mediapipe as mp

class Hands:
    
    def __init__(self, callback, camera_id=0):

        # Tools for the hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

        # Data
        self.is_right_hand_visible = False
        self.is_left_hand_visible = False
        self.hand_order = [0, 1]
        self.velocity_l_x = 0
        self.velocity_l_y = 0
        self.velocity_r_x = 0
        self.velocity_r_y = 0
        self.p0_l_x = 0
        self.p0_l_y = 0
        self.p0_r_x = 0
        self.p0_r_y = 0

        # Configs
        self.show_preview = True

        # Start camera
        self.prev_results = None
        self.cap = cv2.VideoCapture(camera_id) # Cap is the camera

        with self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

            while self.cap.isOpened():
                success, image = self.cap.read() # gets the frame

                if not success:
                    print("Ignoring empty camera frame.")
                    # If loading a video, use 'break' instead of 'continue'.
                    continue

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image) # results

                # Determine what hands are visible
                self.is_right_hand_visible = False
                self.is_left_hand_visible = False


                # if results.multi_handedness:
                #     self.hand_order = []
                #     for hand in results.multi_handedness:

                #         for c in hand.classification:
                #             if c.index == 0: self.is_right_hand_visible = True
                #             if c.index == 1: self.is_left_hand_visible = True
                #             self.hand_order.append(c.index)

                if results.multi_handedness:
                    self.hand_order = []
                    for hand in results.multi_handedness:
                        
                        for c in hand.classification:
                            if c.index == 0: self.is_right_hand_visible = True
                            if c.index == 1: self.is_left_hand_visible = True
                            self.hand_order.append(c.index)
                    
                
                # Compute velocities
                if results.multi_hand_landmarks:
                    for h in range(len(results.multi_hand_landmarks)):
                        hand = results.multi_hand_landmarks[h]

                        if self.is_left_hand_visible and self.is_right_hand_visible:
                            right_hand_order = self.hand_order.index(1)
                            if (h == right_hand_order):

                                # Right hand
                                self.velocity_r_x = (hand.landmark[0].x - self.p0_r_x)
                                self.velocity_r_y = (hand.landmark[0].y - self.p0_r_y)
                                self.p0_r_x = hand.landmark[0].x
                                self.p0_r_y = hand.landmark[0].y
                            
                            else:

                                # Left hand
                                self.velocity_l_x = (hand.landmark[0].x - self.p0_l_x)
                                self.velocity_l_y = (hand.landmark[0].y - self.p0_l_y)
                                self.p0_l_x = hand.landmark[0].x
                                self.p0_l_y = hand.landmark[0].y

                        elif self.is_left_hand_visible:
                            self.velocity_l_x = (hand.landmark[0].x - self.p0_l_x)
                            self.velocity_l_y = (hand.landmark[0].y - self.p0_l_y)
                            self.p0_l_x = hand.landmark[0].x
                            self.p0_l_y = hand.landmark[0].y

                        elif self.is_right_hand_visible:
                            self.velocity_r_x = (hand.landmark[0].x - self.p0_r_x)
                            self.velocity_r_y = (hand.landmark[0].y - self.p0_r_y)
                            self.p0_r_x = hand.landmark[0].x
                            self.p0_r_y = hand.landmark[0].y

                        # print(hand.landmark[0])
                    # self.velocity_x = results.multi_hand_world_landmarks[0]
                    # self.velocity_y = results.multi_hand_world_landmarks[0]
                    # print(results.multi_hand_world_landmarks)
                    # for hand_landmarks in results.multi_hand_world_landmarks:
                    #     print(hand_landmarks)
                        #print(hand_landmarks.landmark[0])

                # Save results for next frame
                self.prev_results = results

                # Draw the hand annotations on the image.
                image.flags.writeable = True
                # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks and self.show_preview:
                    for hand_landmarks in results.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            image,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing_styles.get_default_hand_landmarks_style(),
                            self.mp_drawing_styles.get_default_hand_connections_style())

                        # add to database
                        

                callback(self, results, cv2.flip(image, 1), self.is_right_hand_visible, self.is_left_hand_visible, {"left": { "x": self.velocity_l_x, "y": self.velocity_l_y }, "right": { "x": self.velocity_r_x, "y": self.velocity_r_y }})
                if cv2.waitKey(5) & 0xFF == 27:
                    self.releaseCapture()
                    break

    def set_preview_enabled(self, enabled):
        self.show_preview = enabled

    def releaseCapture(self):
        self.cap.release()

    def getFrame(self):
        return self.prev_frame

    def getResults(self):
        return self.prev_results