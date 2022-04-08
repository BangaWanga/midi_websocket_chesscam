import cv2
import numpy as np

from Server.chesscam.camera import Camera
from Server.chesscam.centroids import Centroids
from Server.chesscam.overlay import Overlay


class ChessCam:
    def __init__(self):
        self.overlay_scale = 0.5
        self.overlay_pos = (0, 0)
        self.set_move_size = 5

        self.grid = np.zeros((8, 8, 2), dtype=np.int32)

        self.camera = Camera()

        self.frame = self.camera.capture_frame_from_videostream()

        self.centroids = Centroids()
        self.overlay = Overlay(self.camera.get_cam_resolution())

        self.grid_captured = False

        self.capture_new_sequence = False #Flag for new sequences
        self.new_sequence_captured = False

        # define color boundaries (lower, upper) (in RGB, since we always flip the frame)
        self.colorBoundaries = [
            [np.array([10, 10, 10]), np.array([255, 56, 50])],  # red
            [np.array([0, 70, 5]), np.array([50, 200, 50])],   # green
            [np.array([4, 31, 86]), np.array([50, 88, 220])]    # blue
        ]
        self.states = np.zeros(self.grid.shape[:2], dtype=np.int32)   # array that holds a persistent state of the chessboard
        print("Chesscam init finished")


    def update(self, updateGrid = True):
        self.frame = self.camera.capture_frame_from_videostream()
        gray_scaled = self.camera.apply_gray_filter(self.frame)

        self.centroids.do_stoff_with_centroids(gray_scaled, updateGrid)

        self.overlay.change_drawing_options(self.overlay_pos, self.overlay_scale)
        img = self.overlay.draw_grid(self.frame)

        # Display the resulting frame
        cv2.imshow('computer visions', img)
        self.process_key_input()

    def run(self, user_trigger=False):
        #At first we need the grid
        if not self.grid_captured:
            self.update(updateGrid=True)
        else:
            self.update(updateGrid=False)
            if user_trigger:
                # ToDo: Do we really need user_trigger?
                pass # actually this is the beat capturing

    def update(self, updateGrid = True):
        self.frame = self.camera.capture_frame_from_videostream()
        gray_scaled = self.camera.apply_gray_filter(self.frame)

        self.centroids.do_stoff_with_centroids(gray_scaled, updateGrid)

        img = self.overlay.draw_grid(self.frame)
        # Display the resulting frame
        cv2.imshow('computer visions', img)
        self.process_key_input()

    def process_key_input(self):
        key = cv2.waitKey(1)
        if key == 113 or key == 27:
            self.quit()

        if key == 97:
            self.overlay_pos = (self.overlay_pos[0] - self.set_move_size, self.overlay_pos[1])
        if key == 100:
            self.overlay_pos = (self.overlay_pos[0] + self.set_move_size, self.overlay_pos[1])
        if key == 119:
            self.overlay_pos = (self.overlay_pos[0], self.overlay_pos[1] - self.set_move_size)
        if key == 115:
            self.overlay_pos = (self.overlay_pos[0], self.overlay_pos[1] + self.set_move_size)

        if key == 43:
            self.overlay_scale += 0.01
        if key == 45:
            self.overlay_scale -= 0.01

        if key != -1:
            print(key)

    def gridToState(self):
        aoiHalfWidth = 5  # half width in pixels of the square area of interest around the centroids
        colored_threshold = 50  # threshold for detecting if a field is colored (measured values are between 0 and 255)

        self.grid = self.grid.astype(np.int32)
        for y in range(8):  # loop over y-coordinate
            for x in range(8):  # loop over y-coordinate
                try:
                    color_state = 0  # initially, color_state is Off (1: red, 2: green, 3: blue)
                    # now loop through the colors to see if there is a significant amount of any
                    # At the end, color_state will always correspond to the last color that was found
                    for colorNum, (lower, upper) in enumerate(self.colorBoundaries):
                        areaOfInterest = self.defineAreaOfInterest(aoiHalfWidth, x, y)

                        mask = cv2.inRange(areaOfInterest, lower, upper)  # returns binary mask: pixels which fall in the range are white (255), others black (0)
                        if np.mean(mask) > colored_threshold:  # if some significant amount of pixels in the mask is 255, we consider it colored
                            color_state = colorNum + 1  # +1 because colorNum is zero-based, but color_state zero is Off
                    self.states[x, y] = color_state

                except (IndexError, cv2.error) as e:
                    # if an error occurs due to invalid coordinates, just don't change the color_state
                    pass

        # dissect the board into the four 16-step sequences (two rows for each sequence of 16 steps)
        return self.states

    def defineAreaOfInterest(self, aoiHalfWidth, x, y):
        # square around field midpoint

        lowerY, upperY = self.grid[x, y, 1] - aoiHalfWidth, self.grid[x, y, 1] + aoiHalfWidth
        lowerX, upperX = self.grid[x, y, 0] - aoiHalfWidth, self.grid[x, y, 0] + aoiHalfWidth
        areaOfInterest = self.frame[lowerY:upperY, lowerX:upperX]
        return areaOfInterest

    def quit(self):
        # When everything done, release the capture
        self.camera.cam.release()
        cv2.destroyAllWindows()
        quit()


if __name__ == "__main__":
    cam = ChessCam()

    while not cam.grid_captured:
        cam.update(True)
    i = 0
    while True:
        if i % 100 == 0:
            pass
        if i > 100:
            break
        cam.run(user_trigger=True)