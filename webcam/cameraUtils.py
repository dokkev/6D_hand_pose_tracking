import cv2
import os
def camera_index():

	MAX_RANGE = 10

	active_camera_indices = []

	for i in range(MAX_RANGE):

		cam = cv2.VideoCapture(i)

		if cam.isOpened():
			active_camera_indices.append(i) # Note that the current index is a functioning camera

	os.system("clear")
	return active_camera_indices