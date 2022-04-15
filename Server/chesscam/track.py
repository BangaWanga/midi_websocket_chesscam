import numpy as np


class Centroids:
    def __init__(self):
        self.grid = np.zeros((8, 8, 2), dtype=np.int32)
        self.states = np.zeros(self.grid.shape[:2], dtype=np.int32)   # array that holds a persistent state of the chessboard
        # define color boundaries (lower, upper) (in RGB, since we always flip the frame)
        self.colorBoundaries = [
            [np.array([10, 10, 10]), np.array([255, 56, 50])],  # red
            [np.array([0, 70, 5]), np.array([50, 200, 50])],   # green
            [np.array([4, 31, 86]), np.array([50, 88, 220])]    # blue
        ]
