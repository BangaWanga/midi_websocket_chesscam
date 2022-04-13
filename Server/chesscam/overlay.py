import numpy as np
import math
import itertools
import cv2
from typing import Tuple
import random


class Overlay:
    def __init__(self, frame_shape, width: int = 20 , height: int = 8, offset: Tuple[int, int] = (0, 0),
                 scale: float = 1.):
        self.frame_shape = frame_shape
        print(frame_shape)
        self.width = width
        self.height = height
        self.frame_width = int(self.frame_shape[1])
        self.frame_height = int(self.frame_shape[0])
        self.offset = offset
        self.scale = scale
        self.grid = self.make_grid()

    def make_grid(self):
        grid = {}
        for i in range(self.height):
            for j in range(self.width):

                random_col = tuple(random.randint(0, 255) for _ in range(3))
                edge0, edge1, edge2, edge3 = self.get_pixel_edges(j, i, self.offset, self.scale)
                line_coordinates = self.get_start_and_endpoints_from_edges(edge0, edge1, edge2, edge3)
                center_point = self.find_center_of_square(edge0, edge1, edge2, edge3)
                grid[(i, j)] = {"line_coordinates": line_coordinates, "center_point": center_point, "color": random_col}
        return grid

    def draw_line(self, img, start=(0, 0), end=(100, 100), line_thickness=2, col=(0, 255, 0)):
        img = img.copy()
        cv2.line(img, start, end, col, thickness=line_thickness)
        return img

    def draw_rectangle(self, img, pts1=(0, 0), pts2=(100, 100), col=(0, 0, 0)):
        img = img.copy()
        cv2.rectangle(img, pts1, pts2, color=col, thickness=3)
        return img

    def draw_circle(self, img, center=(0, 0), radius: int = 2, col=(0, 0, 0)):
        img = img.copy()
        cv2.circle(img, center, radius, color=col, thickness=1, lineType=8, shift=0)
        return img

    def get_pixel_edges(self, x: int, y: int, offset: Tuple[int, int], scale: float):
        assert x <= self.width and y <= self.height
        edge0 = (x * (self.frame_width / self.width), y * (self.frame_height / self.height))
        edge1 = ((x + 1) * (self.frame_width / self.width), y * (self.frame_height / self.height))
        edge2 = (x * (self.frame_width / self.width), (y + 1) * (self.frame_height / self.height))
        edge3 = ((x + 1) * (self.frame_width / self.width), ((y + 1) * (self.frame_height / self.height)))
        edges = (edge0, edge1, edge2, edge3)
        apply_offset = lambda tu: (tu[0] + offset[0], tu[1] + offset[1])
        apply_scale = lambda tu: (tu[0] * scale, tu[1] * scale)
        cast_to_int = lambda tu: (int(tu[0]), int(tu[1]))
        edges = tuple(map(apply_offset, edges))
        edges = tuple(map(apply_scale, edges))
        edges = tuple(map(cast_to_int, edges))
        return edges

    def change_drawing_options(self, offset: Tuple[int, int] = (0, 0), scale: float = 1.):
        self.offset = offset
        self.scale = scale
        self.grid = self.make_grid()

    def draw_grid(self, img):
        for k, v in self.grid.items():
            line_coordinates = v["line_coordinates"]
            for line in line_coordinates:
                startpoint, endpoint = line
                img = self.draw_line(img, startpoint, endpoint, col=v["color"])
            # draw circle in center:
            img = self.draw_circle(img, v["center_point"], col=v["color"])
            if k == (0, 0):
                self.scan_square(img, v["center_point"], debug=True)
            else:
                self.scan_square(img, v["center_point"])

            """

            for line in (line_coordinates[0], line_coordinates[-1]):
                startpoint, endpoint = line
                img = self.draw_rectangle(img, startpoint, endpoint, col=v["color"])
            """
        return img

    def get_start_and_endpoints_from_edges(self, edge0, edge1, edge2, edge3):
        return [(edge0, edge1), (edge0, edge2), (edge2, edge3), (edge1, edge3)]

    @staticmethod
    def calc_diameter(point0, point1):
        return math.sqrt(abs(point1[0] - point0[0])) + math.sqrt(abs(point1[1] - point0[1]))

    def find_center_of_square(self, edge0, edge1, edge2, edge3):
        edges = [edge0, edge1, edge2, edge3]
        comb = list(itertools.combinations(edges, 2))   # every possible combination of the edges
        diam = [Overlay.calc_diameter(*pts) for pts in comb]
        pt0, pt1 = comb[diam.index(max(diam))]  # find points with biggest diameter
        center_point = (int((pt0[0] + pt1[0]) / 2), int((pt0[1] + pt1[1]) / 2))
        return center_point

    @staticmethod
    def scan_square(img, square_center: Tuple[int, int], scan_width: int = 2, debug=False):
        s0 = square_center[0] - int(scan_width/2)
        e0 = square_center[0] + int(scan_width/2)
        s1 = square_center[1] - int(scan_width/2)
        e1 = square_center[1] + int(scan_width/2)
        region = img[s1:e1, s0:e0]
        color_value = np.mean(region, axis=1).mean(axis=0)
        if debug:
            print(f"colorValue: Red {color_value[0]} Green {color_value[1]} Blue {color_value[2]}", )


