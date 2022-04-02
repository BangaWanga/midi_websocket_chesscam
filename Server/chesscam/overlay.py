import cv2
import numpy as np


class Overlay:
    def __init__(self, frame_shape):
        self.frame_shape = frame_shape

    def draw_line(self, img, start=(0, 0), end=(100, 100), line_thickness=2, col=(0, 255, 0)):
        img = img.copy()
        cv2.line(img, start, end, col, thickness=line_thickness)
        return img

    def draw_rectangle(self, img, pts1=(0, 0), pts2=(100, 100)):
        cv2.rectangle(img, pts1, pts2,color=(0, 0, 0), thickness=3)

    def draw_grid(self, img, offset=(0, 0), scale=0.5):
        lines = 8
        width = int(self.frame_shape[0])
        height = int(self.frame_shape[1])

        start_positions_horiontal = np.column_stack(((np.zeros(lines, dtype=int), (np.arange(lines) * height / lines + 1).astype(int))))
        start_positions_vertical = np.column_stack(((np.arange(lines) * height / lines).astype(int), np.zeros(lines, dtype=int)))

        end_positions_horizontal = np.column_stack((np.ones(lines, dtype=int) * height, (np.arange(lines) * height / lines).astype(int)))
        end_positions_vertical = np.column_stack(((np.arange(lines) * width / lines).astype(int), np.ones(lines, dtype=int) * width))
        sp = np.vstack((start_positions_horiontal, start_positions_vertical))
        ep = np.vstack((end_positions_horizontal, end_positions_vertical))
        sp = (sp * scale).astype(int)
        ep = (ep * scale).astype(int)
        sp[..., 0] = sp[..., 0] + offset[0]
        ep[..., 0] = ep[..., 0] + offset[0]
        sp[..., 1] = ep[..., 1] + offset[1]
        ep[..., 1] = ep[..., 1] + offset[1]

        sp = [tuple(pos) for pos in sp]
        ep = [tuple(pos) for pos in ep]

        for i in range(len(sp)):
            img = self.draw_line(img, sp[i], ep[i])
        return img
