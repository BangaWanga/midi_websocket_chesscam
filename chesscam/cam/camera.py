import cv2
import numpy as np


class Camera:
    def __init__(self):
        self.cam = cv2.VideoCapture(0)
    # ToDo: Add method to save color from camera

    def capture_frame_from_videostream(self):
        ret, frame = self.cam.read()
        if ret:
            return self.flip(frame)
        else:
            raise ValueError("Can't read frame")

    def get_cam_resolution(self, ):
        # broken for mac camera, just return static value
        # ToDo: Move to config
        frame = self.capture_frame_from_videostream()
        return frame.shape[0], frame.shape[1]

    def flip(self, frame):
        # flip it since conventions in cv2 are the other way round
        frame = np.flip(frame, axis=1)
        frame = np.flip(frame, axis=2)
        return frame

    def apply_gray_filter(self, img, white_areas=(5, 5)):   # Small number = small white areas
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        # apply moving average
        # this is important to get rid of image noise and make the boundaries between black and white wider
        # the latter leads to smaller white areas after thresholding (see below)
        gray = cv2.blur(gray, white_areas)

        # threshold filter -> convert into 2-color image
        ret, dst = cv2.threshold(gray, 0.6 * gray.max(), 255, 0)
        dst = np.uint8(dst)

        return dst
