import cv2
import numpy as np


class Centroids:
    def __init__(self):
        self.grid = np.zeros((8, 8, 2), dtype=np.int32)

    def do_stoff_with_centroids(self, gray_scaled, updateGrid):
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
        centroids = centroids[np.argsort(centroids[:, 1])]
        if updateGrid:
            try:
                self.grid = self.make_grid(centroids)
                self.grid_captured = True
            except ValueError as e:
                print(e)
        # Write coordinates to the screen
        self.update_centroid_labels(gray_scaled)
        # add rectangle
        # self.draw_rectangle(gray_scaled)
        # img = self.draw_line(img, start=(0, 0), end=self.frame_shape[:2])

    def update_centroid_labels(self, img):
        for i in range(8):
            for j in range(8):
                isBlackField = ((i % 2 == 0) and (j % 2 == 1)) or ((i % 2 == 1) and (j % 2 == 0))
                c = (255 * isBlackField, 255 * isBlackField, 255 * isBlackField)
                cv2.putText(img, "({0}, {1})".format(i, j), tuple(self.grid[i, j]), fontScale=0.2,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            color=c)


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
            white_fields = centroids[y * 4:y * 4 + 4]  # get the 4 centroids of row y
            white_fields = white_fields[np.argsort(white_fields[:, 0])]  # sort them by their x-coordinate
            for x in range(4):
                grid[x * 2 + isodd, y] = white_fields[
                    x]  # have to shift the white field's x-index by 1 in the odd rows

        # Calculate black field pixel coordinates
        for y in range(8):
            isodd = y % 2
            if isodd:
                for x in range(1, 4):
                    self.grid[2 * x, y] = (self.grid[2 * x - 1, y] + self.grid[
                        2 * x + 1, y]) / 2.  # mean position of neighboring white fields
                self.grid[0, y] = self.grid[1, y] - (self.grid[2, y] - self.grid[
                    1, y])  # get leftmost (pos. 0) black field by subtracting the vector pointing from the white field at pos. 1 to the black field at pos. 2 from the pos. of the white field at pos. 1
            else:
                for x in range(3):
                    self.grid[2 * x + 1, y] = (self.grid[2 * x, y] + self.grid[
                        2 * x + 2, y]) / 2.  # mean position of neighboring white fields
                self.grid[7, y] = self.grid[6, y] + (self.grid[6, y] - self.grid[
                    5, y])  # get rightmost (pos. 7) black field by adding the vector pointing from the black field at pos. 5 to the white field at pos. 6 to the pos. of the white field at pos. 6

        # Cast all the coordinates to int (effectively applying the floor function) to yield actual pixel coordinates
        grid = self.grid.astype(np.int32)
        grid = self.grid.astype(np.int32)
        return grid