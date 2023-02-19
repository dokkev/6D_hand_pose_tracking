from cameraUtils import camera_index
from mpl_toolkits.mplot3d import axes3d, Axes3D
import matplotlib.pyplot as plt
import concurrent.futures
import numpy as np
import mediapipe
import cv2
import os

def landmark_cords(landmark, joint_index, scale=100):

	x = landmark[joint_index].x * 100
	y = landmark[joint_index].y * 100
	z = landmark[joint_index].z * 100

	return x, y, z

class Connector:
	def __init__(self, start, end):
		self.start = start
		self.end = end

class Hand3D:

	def __init__(self):

		self.joints  = []

		self.connections = []

		self.connected_to = []

		self.point_color = "#1f77b4"

		self.connector_color = "#d62728"

		for n in range(21):

			if n == 0:

				t1 = Connector(0, 1)
				t2 = Connector(0, 5)
				t3 = Connector(0, 17)

				self.connected_to.append(t1)
				self.connected_to.append(t2)
				self.connected_to.append(t3)

			elif n == 4:
				pass # No connections, end of thumb

			elif n == 8:
				pass # No connections, end of index finger

			elif n == 12:
				pass # No connections, end of middle finger

			elif n == 16:
				pass # No connections, end of ring finger

			elif n == 20:
				pass # No connections, end of pinkie finger

			elif n == 5:
				t1 = Connector(5, 6)
				t2 = Connector(5, 9)

				self.connected_to.append(t1)
				self.connected_to.append(t2)

			elif n == 9:
				t1 = Connector(9, 10)
				t2 = Connector(9, 13)

				self.connected_to.append(t1)
				self.connected_to.append(t2)

			elif n == 13:
				t1 = Connector(13, 14)
				t2 = Connector(13, 17)
	
				self.connected_to.append(t1)
				self.connected_to.append(t2)
			else:
				t = Connector(n, n+1)

				self.connected_to.append(t)

	def _init_joints(self, land_marks):

		for landMarks in land_marks:
			for j in range(21): # For every joint in each and (21 marked)

				x, y, z = landmark_cords(landMarks.landmark, j)

				self.joints.append(ax.scatter(x,y,z, c=self.point_color))
				self.joints[j].x = x
				self.joints[j].y = y
				self.joints[j].z = z

	def _init_connections(self):

		for connector in self.connected_to:
			start = self.joints[connector.start]
			end = self.joints[connector.end]

			x_pairs = [start.x, end.x]
			y_pairs = [start.y, end.y]
			z_pairs = [start.z, end.z]

			self.connections.append(ax.plot(x_pairs, y_pairs, z_pairs, c=self.connector_color))

	def update_joint(self, joint_index, new_point):

		temp = self.joints[joint_index]

		self.joints[joint_index] = new_point
		temp.remove()

	def batch_update_connectors(self):

		c = 0
		for connector in self.connected_to:

			start = self.joints[connector.start]
			end = self.joints[connector.end]

			x_pairs = [start.x, end.x]
			y_pairs = [start.y, end.y]
			z_pairs = [start.z, end.z]

			self.connections[c][0].remove()
			self.connections[c] = ax.plot(x_pairs, y_pairs, z_pairs, c=self.connector_color)

			c += 1


	def batch_update_joints(self, land_marks):
		for landMarks in land_marks:

			for j in range(21):

				x, y, z = landmark_cords(landMarks.landmark, j)

				self.joints[j].remove()
				self.joints[j] = ax.scatter(x, y, z,c=self.point_color)
	
				self.joints[j].x = x
				self.joints[j].y = y
				self.joints[j].z = z

def plot_update(relim=True,autoscale=True,draw=True,show=True):

	if relim is True:
		ax.relim()

	if autoscale is True:
		ax.autoscale_view(True, True, True)

	if draw is True:
		plt.draw()

	if show is True:
		plt.show()

	plt.pause(0.0000000001)

def update_point_ar(point_ar, old_point, new_point):

	count = 0 
	for point in point_ar:

		if point == old_point:
			break

		count += 1

	point_ar[count] = new_point
			
handsModule = mediapipe.solutions.hands

camera = cv2.VideoCapture(0)

plt.ion()
fig = plt.figure()
ax = Axes3D(fig)

ax.set_xlim([0,100])
ax.set_ylim([0,100])
ax.set_zlim([0,100])

ax.set_ylabel("Y axis")
ax.set_xlabel("X axis")
ax.set_zlabel("Z axis")

first = True

with handsModule.Hands(static_image_mode=False, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2) as hands:

	ret, frame = camera.read()

	results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

	HandPoints = Hand3D()

	while True:

		ret, frame = camera.read()
		results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

		if results.multi_hand_landmarks != None and first:

			HandPoints._init_joints(results.multi_hand_landmarks)
			HandPoints._init_connections()
			plot_update()
			first = False

		elif results.multi_hand_landmarks != None and not first:

			HandPoints.batch_update_joints(results.multi_hand_landmarks)
			HandPoints.batch_update_connectors()
			plot_update()

		#ax.plot([0,0],[0,50],[0,100])
		cv2.imshow("Hand Tracking", frame)

		if cv2.waitKey(1) == 27:
			break

camera.release()