import cv2
import np as np


class Overlay:
    def __init__(self, frame_shape):
        self.frame_shape = frame_shape

    def draw_line(self, img, start=(0, 0), end=(100, 100), line_thickness=2, col=(0, 255, 0)):
        img = img.copy()
        import cv2
        cv2.line(img, start, end, col, thickness=line_thickness)
        return img

    def draw_rectangle(self, img, pts1=(0, 0), pts2=(100, 100)):
        cv2.rectangle(img, pts1, pts2,color=(0, 0, 0), thickness=3)

    def draw_grid(self, img, offset=(0, 0), scale=0.5):
        lines = 8
        width = int(self.frame_shape[0] * scale)
        height = int(self.frame_shape[1] * scale)
        start_positions_horiontal = list(
            zip(np.zeros(lines, dtype=int), (np.arange(lines) * height / lines).astype(int)))
        start_positions_vertical = list(
            zip((np.arange(lines) * height / lines).astype(int), np.zeros(lines, dtype=int)))

        end_positions_horizontal = list(
            zip(np.ones(lines, dtype=int) * height, (np.arange(lines) * height / lines).astype(int)))
        end_positions_vertical = list(
            zip((np.arange(lines) * height / lines).astype(int), np.ones(lines, dtype=int) * height))
        print(self.frame_shape)
        sp = start_positions_horiontal + start_positions_vertical + [(width, 0), (0, height)]
        ep = end_positions_horizontal + end_positions_vertical + [(width, height), (width, height)]
        print(sp)
        print(ep)
        for i in range(len(sp)):
            img = self.draw_line(img, sp[i], ep[i])
        return img