import cv2
import numpy as np
from typing import Tuple
import random

class Overlay:
    def __init__(self, frame_shape, width: int = 8, height: int = 8):
        self.frame_shape = frame_shape
        print(frame_shape)
        self.lines = 8
        self.width = width
        self.height = height
        self.frame_width = int(self.frame_shape[0])
        self.frame_height = int(self.frame_shape[1])

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

    def draw_grid(self, img, offset=(0, 0), scale=1.):
        for i in range(self.height):
            for j in range(self.width):
                random_col = tuple(random.randint(0,255) for _ in range(3))
                edge0, edge1, edge2, edge3 = self.get_pixel_edges(i, j, offset, scale)
                for line_coordinates in self.get_start_and_endpoints_from_edges(edge0, edge1, edge2, edge3):
                    startpoint, endpoint = line_coordinates
                    print(f"line_coordinates: {line_coordinates}. startpoint: {startpoint}, endpoint: {endpoint}")
                    img = self.draw_line(img, startpoint, endpoint, col=random_col)
        return img

    def draw_grid_deprecated(self, img, offset=(0, 0), scale=1.):
        sp = [tuple(pos) for pos in self.get_startpoints(offset, scale)]
        ep = [tuple(pos) for pos in self.get_endpoints(offset, scale)]

        for i in range(len(sp)):
            img = self.draw_line(img, sp[i], ep[i])
        return img

    def get_start_and_endpoints_from_edges(self, edge0, edge1, edge2, edge3):
        return [(edge0, edge1), (edge0, edge2), (edge2, edge3), (edge1, edge3)]

    def get_endpoints(self, offset, scale):
        ep = np.vstack((self.get_ep_horizontal(), self.get_ep_vertical()))
        ep = self.scale_and_offset(offset, scale, ep)
        return ep

    def get_startpoints(self, offset, scale):
        sp = np.vstack((self.get_sp_horizontal(), self.get_sp_vertical()))
        sp = self.scale_and_offset(offset, scale, sp)
        return sp

    def get_ep_vertical(self):
        end_positions_vertical = np.column_stack(
            (((np.arange(self.width) / self.width) * self.frame_width).astype(int), np.ones(self.width, dtype=int) * self.frame_width))
        return end_positions_vertical

    def get_ep_horizontal(self):
        end_positions_horizontal = np.column_stack(
            (np.ones(self.height, dtype=int) * self.frame_height, (np.arange(self.height) * self.frame_height).astype(int)))
        return end_positions_horizontal

    def get_sp_vertical(self):
        start_positions_vertical = np.column_stack(
            ((np.arange(self.lines) * self.frame_height / self.lines).astype(int), np.zeros(self.lines, dtype=int)))
        return start_positions_vertical

    def get_sp_horizontal(self):
        start_positions_horiontal = np.column_stack(
            (np.zeros(self.lines, dtype=int), (np.arange(self.lines) * self.frame_height / self.lines).astype(int)))
        return start_positions_horiontal

    def scale_and_offset(self, offset, scale, sp):
        sp = (sp * scale).astype(int)
        sp[..., 0] = sp[..., 0] + offset[0]
        sp[..., 1] = sp[..., 1] + offset[1]
        return sp
