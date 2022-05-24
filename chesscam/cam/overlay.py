import numpy as np
import itertools
from typing import Tuple
import typing
from enum import Enum
from chesscam.cam.color_predictor import NearestNeighbour


class DisplayOption(Enum):
    Normal = 0
    Calibrate = 1


class Overlay:
    def __init__(self, frame_shape, width: int = 8, height: int = 8, offset: Tuple[int, int] = (0, 0),
                 scale: float = 1.):
        self.colors = ("None", "green", "red", "blue", "yellow")
        self.width = width
        self.height = height
        self.frame_height, self.frame_width = frame_shape
        self.offset = offset
        self.scale = scale
        self.display_option = DisplayOption.Calibrate
        self.color_predictor = NearestNeighbour(colors=self.colors)
        self.cursor_field = (0, 0)  # ToDo: Less variables for cursor
        self.cursor = np.array([0., 0.])
        self.cursor_absolute = (0., 0.)
        self.selected_color = None
        self.calibrate_field = False  # If this value is tuple[int, int], the selected field is calibrated with selected color
        # flattened grid_positions [(0,0), (0, 1), ...]
        self.grid_positions = list(itertools.chain(*[[(i, j) for j in range(self.width)] for i in range(self.height)]))
        self.grid_positions = np.array(self.grid_positions)

        self._grid = np.zeros(shape=(self.width, self.height, len(self.colors)), dtype=int)
        self.color_buffer = 5  # how many concurrent frames a color can be guessed

    @property
    def chess_board_values(self) -> dict:
        col_classes = np.argmax(self._grid, axis=2).flatten()
        return {int(pos): int(col_classes[pos]) for pos in np.argwhere(col_classes != 0)}

    @property
    def rect_width(self):
        return int((self.frame_width * self.scale) / self.width)

    @property
    def rect_height(self):
        return int((self.frame_height * self.scale) / self.height)

    def select_field(self, color_class: int):
        self.selected_color = color_class
        self.calibrate_field = True

    def get_rect_start_position(self, position: Tuple[int, int]):  # left upper corner
        return int((self.frame_width / self.width) * position[0]), int((self.frame_height / self.height) * position[1])

    def center_point_from_grid_position(self, position) -> Tuple[int, int]:
        return int((position[0] + .5) * self.rect_width), int((position[1] + .5) * self.rect_height)

    def calibrate(self, frame, positions: typing.List[tuple], selected_colors: typing.List[int]):
        # rgb_values = self.color_scan(frame)
        measured_colors = [self.get_square_color(frame, p) for p in positions]
        self.color_predictor.add_samples(selected_colors, measured_colors)
        # self.calibrate_field = False  # ToDo: Maybe add more samples right away?
        # self.color_predictor.add_sample(self.selected_color, color)

    def update_color_values(self, frame, error_threshold: float = 1.3):  # ToDo: Make this all pure numpy
        colors = self.color_scan(frame)
        for i, position in enumerate(self.grid_positions):
            color_class, error = self.color_predictor.predict_color(colors[i])
            if error == -1 or error > error_threshold:
                self._grid[position[0]][position[1]] = self._grid[position[0]][position[1]] - 1
            else:
                diff = np.array([-1 for _ in self.colors])
                diff[color_class] = 1
                self._grid[position[0]][position[1]] += diff
            self._grid[position[0]][position[1]] = np.clip(self._grid[position[0]][position[1]], 0,
                                                           self.color_buffer)

    def get_current_color_class(self, position: Tuple[int, int]):
        min_count = max(1, int(self.color_buffer / 2))  # how many times a color has to be counted before it is valid
        if (self._grid[position[0]][position[1]] < min_count).all():
            return None
        return int(np.argmax(self._grid[position[0]][position[1]]))

    def color_scan(self, frame: np.ndarray):
        x_from = np.array((self.frame_width * self.grid_positions[..., 0] / self.width) + self.offset[0]) * self.scale
        x_to = np.array(
            ((self.frame_width * (self.grid_positions[..., 0] + 1)) / self.width) + self.offset[0]) * self.scale
        y_from = np.array(
            ((self.frame_height * self.grid_positions[..., 1]) / self.height) + self.offset[1]) * self.scale
        y_to = np.array(
            ((self.frame_height * (self.grid_positions[..., 1] + 1)) / self.height) + self.offset[1]) * self.scale
        y_from, y_to, x_from, x_to = y_from.astype(int), y_to.astype(int), x_from.astype(int), x_to.astype(int)
        rgb_values = np.zeros(shape=(64, 3))

        reduce_color = lambda x: np.mean(x, axis=(0, 1))  # map area of pixels to single rgb-value
        rgb_values[...] = np.stack([
            reduce_color(frame[y_from[i]:y_to[i], x_from[i]:x_to[i]])
            for i in range(64)
        ])
        return rgb_values

    def get_square_color(self, img, position: Tuple[int, int]):  # ToDo: np.array cast needed?
        x_from = np.array((self.frame_width * position[0] / self.width) + self.offset[0]) * self.scale
        x_to = np.array(((self.frame_width * (position[0] + 1)) / self.width) + self.offset[0]) * self.scale
        y_from = np.array(((self.frame_height * position[1]) / self.height) + self.offset[1]) * self.scale
        y_to = np.array(((self.frame_width * (position[1] + 1)) / self.height) + self.offset[1]) * self.scale
        y_from, y_to, x_from, x_to = y_from.astype(int), y_to.astype(int), x_from.astype(int), x_to.astype(int)
        aoi = img[y_from:y_to, x_from:x_to]  # area of interest
        return np.mean(aoi, axis=1).mean(axis=0)
