import sys
import mediapipe as mp
import pyzed.sl as sl
import numpy as np

class Zed():
    def __init__(self):

        # Decide if SVO or Live
        if len(sys.argv) != 2:
            print("Using Live stream from ZED camera")
            self.input_type = sl.InputType()
            self.svo_mode = False
        else:
            filepath = sys.argv[1]
            print("Reading SVO file: {0}".format(filepath))
            self.input_type = sl.InputType()
            self.input_type.set_from_svo_file(filepath)
            self.svo_mode = True

        # Initialize the ZED camera
        self.zed = sl.Camera()
        self.init_params = sl.InitParameters(input_t=self.input_type)
        self.init_params.camera_resolution = sl.RESOLUTION.HD1080
        self.init_params.camera_fps = 30
        self.init_params.depth_mode = sl.DEPTH_MODE.ULTRA
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.depth_minimum_distance = 0.3
        self.init_params.depth_maximum_distance = 40

 
        # Open the camera
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS :
            print(repr(err))
            self.zed.close()
            exit(1)

        # Create and set RuntimeParameters after opening the camera
        self.runtime_parameters = sl.RuntimeParameters()
        self.runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL  # Use FILL sensing mode
        # Setting the depth confidence parameters
        self.runtime_parameters.confidence_threshold = 100
        self.runtime_parameters.textureness_confidence_threshold = 100

        # Get Camera Calibration Parameters
        self.camera_params = self.zed.get_camera_information().calibration_parameters.left_cam
        self.fx = self.camera_params.fx  # Focal length in pixels (x-axis)
        self.fy = self.camera_params.fy  # Focal length in pixels (y-axis)
        self.cx = self.camera_params.cx  # X-coordinate of the principal point
        self.cy = self.camera_params.cy  # Y-coordinate of the principal point

        


        # declare image, depth, nd point cloud
        self.image = sl.Mat()
        self.depth = sl.Mat()
        self.point_cloud = sl.Mat()

    def print_information(self):
        print("Resolution: {0}, {1}.".format(round(self.zed.get_camera_information().camera_resolution.width, 2), self.zed.get_camera_information().camera_resolution.height))
        print("Camera FPS: {0}".format(self.zed.get_camera_information().camera_fps))
        print("Depth mode: {0}.".format(self.init_params.depth_mode))
        print("Sensing mode: {0}.".format(self.runtime_parameters.sensing_mode))
        if self.svo_mode:
            print("Frame count: {0}.\n".format(self.zed.get_svo_number_of_frames())) 

    def get_image(self):
        # Retrieve left image
        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
        # Retrieve depth map. Depth is aligned on the left image
        self.zed.retrieve_image(self.depth, sl.VIEW.DEPTH)
        # Retrieve colored point cloud. Point cloud is aligned on the left image.
        self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU)

        # convert zed image to numpy array
        self.img = self.image.get_data()
        self.depth_img = self.depth.get_data()