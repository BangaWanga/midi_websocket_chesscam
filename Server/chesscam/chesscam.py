import cv2
import numpy as np

from Server.chesscam.camera import Camera
from Server.chesscam.centroids import Centroids
from Server.chesscam.overlay import Overlay


class ChessCam:
    def __init__(self):

        self.camera = Camera()

        self.frame = self.camera.capture_frame_from_videostream()

        self.overlay = Overlay(self.camera.get_cam_resolution())    # handle scale and pos differently

        self.grid_captured = False


        print("Chesscam init finished")

    def update(self):
        self.frame = self.camera.capture_frame_from_videostream()
        #_gray_scaled = self.camera.apply_gray_filter(self.frame)

        #self.centroids.do_stoff_with_centroids(gray_scaled, updateGrid)

        img = self.overlay.draw_grid(self.frame)
        # Display the resulting frame
        cv2.imshow('computer visions', img)
        self.process_key_input()

    def process_key_input(self):
        key = cv2.waitKey(1)
        if key == 113 or key == 27:
            self.quit()
        move_size = 10
        offset = self.overlay.offset
        scale = self.overlay.scale
        match key:
            case 97:
                offset = (offset[0] - move_size, offset[1])
            case 100:
                offset = (offset[0] + move_size, offset[1])
            case 119:
                offset = (offset[0], offset[1] - move_size)
            case 115:
                offset = (offset[0], offset[1] + move_size)
            case 43:
                scale += 0.01
            case 45:
                scale -= 0.01
            case -1:
                print(key)
        self.overlay.change_drawing_options(offset, scale)


    def quit(self):
        # When everything done, release the capture
        self.camera.cam.release()
        cv2.destroyAllWindows()
        quit()


if __name__ == "__main__":
    cam = ChessCam()

    while True:
        cam.update()
