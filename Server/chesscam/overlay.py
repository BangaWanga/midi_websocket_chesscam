import cv2
import numpy as np


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

    def draw_grid(self, img, offset=(0, 0), scale=0.5):
        sp = [tuple(pos) for pos in self.get_sp(offset, scale)]
        ep = [tuple(pos) for pos in self.get_ep(offset, scale)]

        for i in range(len(sp)):
            img = self.draw_line(img, sp[i], ep[i])
        return img

    def get_ep(self, offset, scale):
        ep = np.vstack((self.get_ep_horizontal(), self.get_ep_vertical()))
        ep = self.scale_and_offset(offset, scale, ep)
        return ep

    def get_sp(self, offset, scale):
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
