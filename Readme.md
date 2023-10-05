<img src="img/demo.gif">

# 3D Hand Landmarks Trakcing with MediaPipe
Google's [MediaPipe](https://google.github.io/mediapipe/) provides a pose estimation model of images and videos of human hands. Along with its capability of palm detection and hand landmark estimation, this project aims to track 21 joints of human hands in the 3D space in real-time. 


## Webcam
MediaPipe is capable of estimating 3D pose without depth perception using 2D-3D lifting through learning a large dataset of hand images and their corresponding 3D pose information. Although the model does not provide precise 3D pose estimation in meter, it can provide a useful estimation of hand pose in the 3D space.

## ZED
Combining with ZED 2i camera, the 3D hand pose tracking can be more accurate with depth perception. The [ZED SDK](https://www.stereolabs.com/developers/release/) provides an API to process depth information of real-time images and record the depth map of a video.
