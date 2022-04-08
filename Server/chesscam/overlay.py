import cv2
import numpy as np
from typing import Tuple
import random


class Overlay:
    def __init__(self, frame_shape, width: int = 8, height: int = 8, offset: Tuple[int, int] = (0, 0), scale: float = 1.):
        self.frame_shape = frame_shape
        print(frame_shape)
        self.lines = 8
        self.width = width
        self.height = height
        self.frame_width = int(self.frame_shape[0])
        self.frame_height = int(self.frame_shape[1])
        self.offset = offset
        self.scale = scale
        self.grid = self.make_grid()

    def draw_line(self, img, start=(0, 0), end=(100, 100), line_thickness=2, col=(0, 255, 0)):
        img = img.copy()
        cv2.line(img, start, end, col, thickness=line_thickness)
        return img

    def draw_rectangle(self, img, pts1=(0, 0), pts2=(100, 100)):
        cv2.rectangle(img, pts1, pts2,color=(0, 0, 0), thickness=3)

    def get_pixel_edges(self, x: int, y: int, offset: Tuple[int, int], scale: float):
        edge0 = ((x / self.width) * self.frame_width, (y / self.height) * self.frame_height)
        edge1 = ((x + 1 / self.width) * self.frame_width, (y / self.height) * self.frame_height)
        edge2 = ((x / self.width) * self.frame_width, (y + 1/ self.height) * self.frame_height)
        edge3 = ((x + 1/ self.width) * self.frame_width, (y + 1 / self.height) * self.frame_height)
        edges = (edge0, edge1, edge2, edge3)
        apply_offset = lambda tu: (tu[0] + offset[0], tu[1] + offset[1])
        apply_scale = lambda tu: (tu[0] * scale, tu[1] * scale)
        cast_to_int = lambda tu: (int(tu[0]), int(tu[1]))
        edges = tuple(map(apply_offset, edges))
        edges = tuple(map(apply_scale, edges))
        edges = tuple(map(cast_to_int, edges))
        return edges

    def make_grid(self):
        grid = {}
        for i in range(self.height):
            for j in range(self.width):
                random_col = tuple(random.randint(0,255) for _ in range(3))
                edge0, edge1, edge2, edge3 = self.get_pixel_edges(i, j, self.offset, self.scale)
                line_coordinates = self.get_start_and_endpoints_from_edges(edge0, edge1, edge2, edge3)
                grid[(i, j)] = {"line_coordinates": line_coordinates, "color": random_col}
        return grid

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
        return img

    def get_start_and_endpoints_from_edges(self, edge0, edge1, edge2, edge3):
        return [(edge0, edge1), (edge0, edge2), (edge2, edge3), (edge1, edge3)]

