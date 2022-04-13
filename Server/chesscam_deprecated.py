import cv2
import numpy as np

from Server.chesscam.camera import Camera
from Server.overlay_deprecated import Overlay


class ChessCam:
    def __init__(self):
        self.grid = np.zeros((8, 8, 2), dtype=np.int32)

        self.camera = Camera()

        self.frame = self.camera.capture_frame_from_videostream()
        self.frame_shape = self.frame.shape

        self.overlay = Overlay(self.frame_shape[:2])

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
        img = self.frame

        # label connected components and calculate the centroids of each chess field
        ret, labels, stats, centroids = cv2.connectedComponentsWithStats(gray_scaled, 4)

        # find the labels of the black components
        # in the end, we want only the white fields
        # also, this helps to remove the big parasitic component which is basically the whole black background with centroid in the middle of the screen
        blackLabels = []
        for label in range(len(centroids)):
            lblIndices = np.where(labels == label)
            if gray_scaled[lblIndices][0] == 0:
                blackLabels.append(label)
        # remove all the centroids of the black components for further processing
        centroids = np.delete(centroids, blackLabels, axis=0)

        # Sorting and rearranging centroids
        # This is to be able to assign the centroids to actual chessboard fields
        # In the physical setup need to make sure that the board axes are quite parallel to the image borders for this to work
        # Trapezoidal tilting should be no problem though
        # This sorts them row-wise from top to bottom (with increasing y-coordinate), but unordered x-coordinate
        centroids = centroids[np.argsort(centroids[:,1])]
        if updateGrid:
            try:
                self.grid = self.make_grid(centroids)
                self.grid_captured = True
                print("Grid Captured.")
            except ValueError as e:
                print(e)

        # Write coordinates to the screen
        self.update_centroid_labels(gray_scaled)

        # add rectangle
        #self.draw_rectangle(gray_scaled)
        #img = self.draw_line(img, start=(0, 0), end=self.frame_shape[:2])

        img = self.overlay.draw_grid(img)

        #img = self.overlay.(img)


        # Display the resulting frame
        cv2.imshow('computer visions', img)
        self.process_input_and_quit()

    def update_centroid_labels(self, img):
        for i in range(8):
            for j in range(8):
                isBlackField = ((i % 2 == 0) and (j % 2 == 1)) or ((i % 2 == 1) and (j % 2 == 0))
                c = (255 * isBlackField, 255 * isBlackField, 255 * isBlackField)
                cv2.putText(img, "({0}, {1})".format(i, j), tuple(self.grid[i, j]), fontScale=0.2,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            color=c)

    def process_input_and_quit(self):
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return

    def make_grid(self, centroids):
        # We assume that the field in the upper left corner is white
        # We should have 32 measured centroids in total (for the white fields)
        # The black ones have to be calculated from the white ones here
        if centroids.shape[0] < 32:
            raise ValueError("Grid needs 32 centroids")
        # Initialize the grid (8x8 array of 2-coordinate arrays)
        grid = np.zeros((8, 8, 2))

        # Fill in the (measured) pixel coordinates of the white fields
        for y in range(8):
            isodd = y % 2
            white_fields = centroids[y*4:y*4+4] # get the 4 centroids of row y
            white_fields = white_fields[np.argsort(white_fields[:, 0])]  # sort them by their x-coordinate
            for x in range(4):
                grid[x*2 + isodd, y] = white_fields[x]  # have to shift the white field's x-index by 1 in the odd rows

        # Calculate black field pixel coordinates
        for y in range(8):
            isodd = y%2
            if isodd:
                for x in range(1,4):
                    self.grid[2*x, y] = (self.grid[2*x - 1, y] + self.grid[2*x + 1, y]) / 2.  # mean position of neighboring white fields
                self.grid[0, y] = self.grid[1, y] - (self.grid[2, y] - self.grid[1, y])  # get leftmost (pos. 0) black field by subtracting the vector pointing from the white field at pos. 1 to the black field at pos. 2 from the pos. of the white field at pos. 1
            else:
                for x in range(3):
                    self.grid[2*x + 1, y] = (self.grid[2*x, y] + self.grid[2*x + 2, y]) / 2.  # mean position of neighboring white fields
                self.grid[7, y] = self.grid[6, y] + (self.grid[6, y] - self.grid[5, y])  # get rightmost (pos. 7) black field by adding the vector pointing from the black field at pos. 5 to the white field at pos. 6 to the pos. of the white field at pos. 6

        # Cast all the coordinates to int (effectively applying the floor function) to yield actual pixel coordinates
        grid = self.grid.astype(np.int32)
        grid = self.grid.astype(np.int32)
        return grid

    def gridToState(self):

        # tolerance = 80
        aoiHalfWidth = 5  # half width in pixels of the square area of interest around the centroids
        colored_threshold = 50  # threshold for detecting if a field is colored (measured values are between 0 and 255)

        self.grid = self.grid.astype(np.int32)
        # states = np.zeros(self.grid.shape[:2], dtype=np.int)
        for y in range(8):  # loop over y-coordinate
            for x in range(8):  # loop over y-coordinate
                try:
                    color_state = 0  # initially, color_state is Off (1: red, 2: green, 3: blue)
                    # now loop through the colors to see if there is a significant amount of any
                    # At the end, color_state will always correspond to the last color that was found
                    for colorNum, (lower, upper) in enumerate(self.colorBoundaries):
                        # define area of interest (square around field midpoint)
                        lowerY, upperY = self.grid[x, y, 1] - aoiHalfWidth, self.grid[x, y, 1] + aoiHalfWidth
                        lowerX, upperX = self.grid[x, y, 0] - aoiHalfWidth, self.grid[x, y, 0] + aoiHalfWidth
                        areaOfInterest = self.frame[lowerY:upperY, lowerX:upperX]

                        mask = cv2.inRange(areaOfInterest, lower, upper)  # returns binary mask: pixels which fall in the range are white (255), others black (0)
                        if np.mean(mask) > colored_threshold:  # if some significant amount of pixels in the mask is 255, we consider it colored
                            color_state = colorNum + 1  # +1 because colorNum is zero-based, but color_state zero is Off
                    self.states[x, y] = color_state

                except (IndexError, cv2.error) as e:
                    # if an error occurs due to invalid coordinates, just don't change the color_state
                    pass

        # dissect the board into the four 16-step sequences (two rows for each sequence of 16 steps)
        print("states Shape: ", self.states.shape)
        return self.states

    def printColors(self, j, i):
        aoiHalfWidth = 2
        areaOfInterest = self.frame[self.grid[j, i, 1]-aoiHalfWidth:self.grid[j, i, 1]+aoiHalfWidth, self.grid[j, i, 0]-aoiHalfWidth:self.grid[j, i, 0]+aoiHalfWidth]
        print(areaOfInterest)

    def setRange(self, colorIndex, j, i):
        aoiHalfWidth = 2
        areaOfInterest = self.frame[self.grid[j, i, 1]-aoiHalfWidth:self.grid[j, i, 1]+aoiHalfWidth, self.grid[j, i, 0]-aoiHalfWidth:self.grid[j, i, 0]+aoiHalfWidth]
        meanColor = np.mean(np.mean(areaOfInterest, axis=0), axis=0)
        lowerColor = np.clip(meanColor - 20, 0, 255).astype(np.uint8)
        upperColor = np.clip(meanColor + 20, 0, 255).astype(np.uint8)

        self.colorBoundaries[colorIndex] = [lowerColor, upperColor]

    def quit(self):
        # When everything done, release the capture
        self.cam.release()
        cv2.destroyAllWindows()

    def save_calibrated(self):
        np.save('colors.yamama', self.colorBoundaries)

    def load_calibrated(self):
        try:
            self.colorBoundaries = np.load('colors.yamama')
        except Exception as _e:
            print("No file detected for color values")

    @staticmethod
    def point_in_triangle(p, v1, v2, v3):
        """Checks whether a point is within the given triangle

        The function checks, whether the given point p is within the triangle defined by the the three corner point v1,
        v2 and v3.
        This is done by checking whether the point is on all three half-planes defined by the three edges of the triangle.
        :param p: The point to be checked (tuple with x any y coordinate)
        :param v1: First vertex of the triangle (tuple with x any y coordinate)
        :param v2: Second vertex of the triangle (tuple with x any y coordinate)
        :param v3: Third vertex of the triangle (tuple with x any y coordinate)
        :return: True if the point is within the triangle, False if not
        """

        def _test(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        b1 = _test(p, v1, v2) < 0.0
        b2 = _test(p, v2, v3) < 0.0
        b3 = _test(p, v3, v1) < 0.0

        return (b1 == b2) and (b2 == b3)


if __name__ == "__main__":
    cam = ChessCam()

    while not cam.grid_captured:
        cam.update(True)
    i = 0
    while True:
        try:
            cam.run(user_trigger=True)
        except:
            pass
    #cam.gridToState()
    cam.quit()