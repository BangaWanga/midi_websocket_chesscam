import numpy as np
import math
import itertools
import cv2
from typing import Tuple
import random
from enum import Enum
from color_predictor import RangeBased

class DisplayOption(Enum):
    Normal = 0
    Calibrate = 1

# ToDo:
# 1. Get picture
# Different procedures: Check colors each n-steps


class Overlay:
    def __init__(self, frame_shape, width: int = 8, height: int = 8, offset: Tuple[int, int] = (0, 0),
                 scale: float = 1.):
        self.colors = ("green", "red", "blue", "yellow")
        self.width = width
        self.height = height
        self.frame_height, self.frame_width = frame_shape
        self.offset = offset
        self.scale = scale
        self.grid = {}
        self.update_grid(False)
        self.display_option = DisplayOption.Calibrate
        self.color_predictor = RangeBased(colors=self.colors)
        self.cursor_field = (0, 0)  # ToDo: Less variables for cursor
        self.cursor = np.array([0., 0.])
        self.cursor_absolute = (0., 0.)
        self.selected_color = None
        self.calibrate_field = False  # If this value is tuple[int, int], the selected field is calibrated with selected color
        # flattened grid_positions [(0,0), (0, 1), ...]
        self.grid_positions = list(itertools.chain(*[[(i, j) for j in range(self.width)] for i in range(self.height)]))

        # experimental purely numpy grid
        self.np_grid = np.zeros(shape=(self.width, self.height, len(self.colors)), dtype=int)
        self.color_buffer = 5   # how many concurrent frames a color can be guessed

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

    def update_grid(self, ignore_colors: bool):
        grid = {}   # ToDo: Pull values from self.grid and only update those that changed
        for i in range(self.height):
            for j in range(self.width):
                random_col = tuple(random.randint(0, 255) for _ in range(3))
                edge0, edge1, edge2, edge3 = self.get_pixel_edges(j, i, self.offset, self.scale)
                line_coordinates = self.get_start_and_endpoints_from_edges(edge0, edge1, edge2, edge3)
                center_point = self.find_center_of_square(edge0, edge1, edge2, edge3)
                if ignore_colors:  # Get old colors to new positions
                    if (i, j) not in self.grid:
                        raise ValueError("Grid is not initalized but ignore colors is False")
                    grid[(i, j)] = {"line_coordinates": line_coordinates,
                                    "center_point": center_point,
                                    "color": self.grid[(i, j)]["color"],
                                    "edges": self.grid[(i, j)]["edges"]}
                else:
                    grid.update(
                        {(i, j):
                             {"line_coordinates": line_coordinates, "center_point": center_point, "color": random_col,
                              "edges": (edge0, edge1, edge2, edge3)
                              }}
                    )
        self.grid = grid

    def scroll_display_option(self):
        options = [d for d in DisplayOption]
        self.display_option = DisplayOption(options[(options.index(self.display_option) + 1) % len(options)])  # sorry

    def draw_line(self, img, start=(0, 0), end=(100, 100), line_thickness=2, col=(0, 255, 0)):
        img = img.copy()
        cv2.line(img, start, end, col, thickness=line_thickness)
        return img

    def draw_rectangle(self, img, pts1=(0, 0), pts2=(100, 100), col=(0, 0, 0)): # ToDo: Enable moving rects with WASD
        cv2.rectangle(img, pts1, pts2, color=col, thickness=3)
        return img

    def draw_circle(self, img, center=(0, 0), radius: int = 2, col=(0, 0, 0)):
        img = img.copy()
        cv2.circle(img, center, radius, color=col, thickness=1, lineType=8, shift=0)
        return img

    def write_text(self, img, text: str, start_pos=(0, 0), font_scale: float = .8, font_color=(255, 255, 255),
                   thickness=1, line_type=1, x_offset: int = 5, y_offset: int = 30):
        img = img.copy()
        start_pos = (start_pos[0] + x_offset, start_pos[1] + y_offset)
        cv2.putText(img, text,
                    start_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    font_color,
                    thickness,
                    line_type)
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
        return edges  # upper_left, upper_right, lower_left, lower_right

    def change_drawing_options(self, offset: Tuple[int, int] = (0, 0), scale: float = 1.):
        self.offset = offset
        self.scale = scale

    def draw_cursor(self, img):
        if self.cursor_field is None:
            return img
        img = img.copy()
        left_upper_corner = self.get_rect_start_position(self.cursor_field)
        right_lower_corner = int(left_upper_corner[0] + self.rect_width), int(left_upper_corner[1] + self.rect_height)
        img = self.draw_rectangle(img, left_upper_corner, right_lower_corner)
        return img

    def draw_grid(self, img, grid_color=(100, 50, 50)):  # ToDo: Make differentiation between display modes earlier
        img = np.ascontiguousarray(img, dtype=np.uint8)
        for pos in self.grid_positions:
            left_upper_corner = self.get_rect_start_position(pos)
            right_lower_corner = int(left_upper_corner[0] + self.rect_width), int(
                left_upper_corner[1] + self.rect_height)
            img = self.draw_rectangle(img, left_upper_corner, right_lower_corner, col=grid_color)
            img = self.draw_classes(img, pos)
        img = self.draw_cursor(img)
        self.update_cursor()
        return img

    def center_point_from_grid_position(self, position) -> Tuple[int, int]:
        return int((position[0] + .5) * self.rect_width), int((position[1] + .5) * self.rect_height)

    def calibrate(self, img, position: tuple, _field_info: dict):
        if self.selected_color is None or not self.calibrate_field:
            return img
        color = self.get_square_color(img, position)
        if position == self.cursor_field:
            print("Adding sample ", color, " for field ", position, " and color ", self.selected_color, self.color_predictor.colors[self.selected_color])
            self.color_predictor.add_sample(self.selected_color, color)
            self.calibrate_field = False    # ToDo: Maybe add more samples right away?
            # self.color_predictor.add_sample(self.selected_color, color)
        return img

    def update_color_values(self, color, position: Tuple[int, int], error_threshold: float = 0.3):
        color_class, error = self.color_predictor.predict_color(color)
        if error == -1 or error > error_threshold:
            self.np_grid[position[0]][position[1]] = self.np_grid[position[0]][position[1]] - 1
        else:
            diff = np.array([-1 for _ in self.colors])
            diff[color_class] = 1
            self.np_grid[position[0]][position[1]] += diff
        self.np_grid[position[0]][position[1]] = np.clip(self.np_grid[position[0]][position[1]], 0, self.color_buffer)

    def get_current_color_class(self, position: Tuple[int, int]):
        min_count = max(1, int(self.color_buffer / 2)) # how many times a color has to be counted before it is valid
        if (self.np_grid[position[0]][position[1]] < min_count).all():
           return None
        return int(np.argmax(self.np_grid[position[0]][position[1]]))

    def draw_classes(self, img, position: Tuple[int, int]):  # TODO: If Calibrate mode, draw error to UI somehow
        color = self.get_square_color(img, position)
        self.update_color_values(color, position)
        col_complementary = self.color_predictor.get_complementary_color(color)
        text_offset = (- int(self.rect_width / 2), - int(self.rect_height / 2))
        center_point = self.center_point_from_grid_position(position)
        center_point = (center_point[0] + text_offset[0], center_point[1] + text_offset[1])

        # error2color = cv2.cvtColor(np.array(error * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB).reshape(3)
        # error2color = tuple(np.array(error2color) / 255)
        # x = f"{str(_error)}"
        color_str = "NONE"
        if not (self.np_grid[position[0]][position[1]] == 0).all():
            color_class = int(np.argmax(self.np_grid[position[0]][position[1]]))
            color_str = self.colors[color_class]

        img = self.write_text(img, f"{color_str}", start_pos=center_point, font_color=col_complementary)
        return img

    def draw_matching_circles(self, img, center_point: (0, 0)):
        col = self.get_square_color(img, center_point)
        img = self.draw_circle(img, center_point, col=col, radius=20)
        return img

    def get_start_and_endpoints_from_edges(self, edge0, edge1, edge2, edge3):
        return [(edge0, edge1), (edge0, edge2), (edge2, edge3), (edge1, edge3)]

    def find_center_of_square(self, edge0, edge1, edge2, edge3):
        edges = [edge0, edge1, edge2, edge3]
        comb = list(itertools.combinations(edges, 2))  # every possible combination of the edges
        diam = [Overlay.calc_diameter(*pts) for pts in comb]
        pt0, pt1 = comb[diam.index(max(diam))]  # find points with biggest diameter
        center_point = (int((pt0[0] + pt1[0]) / 2), int((pt0[1] + pt1[1]) / 2))
        return center_point

    @staticmethod
    def calc_diameter(point0, point1):
        return math.sqrt(abs(point1[0] - point0[0])) + math.sqrt(abs(point1[1] - point0[1]))

    @staticmethod
    def get_region(img, square_center: Tuple[int, int], scan_width: int = 2):
        s0 = square_center[0] - int(scan_width / 2)
        e0 = square_center[0] + int(scan_width / 2)
        s1 = square_center[1] - int(scan_width / 2)
        e1 = square_center[1] + int(scan_width / 2)
        region = img[s1:e1, s0:e0]
        return region

    def update_cursor(self, sensitivity_threshold=.15, movement_speed=0.05):
        add_x = self.cursor[0] if abs(self.cursor[0]) > sensitivity_threshold else 0.
        add_y = self.cursor[1] if abs(self.cursor[1]) > sensitivity_threshold else 0.
        if add_x == 0. and add_y == 0.:
            return
        self.cursor_absolute = (self.cursor_absolute[0] + add_x * movement_speed), (
                    self.cursor_absolute[1] + add_y * movement_speed)
        self.cursor_absolute = min(float(self.width), max(0., self.cursor_absolute[0])), min(float(self.height), max(0.,
                                                                                                                     self.cursor_absolute[
                                                                                                                         1]))
        self.cursor_field = int(self.cursor_absolute[0]), int(self.cursor_absolute[1])

    def move_cursor(self, axis: int, value: float):
        self.cursor[axis % 2] = value  # with % 2 we can address both joysticks
        self.update_cursor()

    def get_square_color(self, img, position: Tuple[int, int]):
        x_from = np.array((self.frame_width * position[0] / self.width) + self.offset[0]) * self.scale
        x_to = np.array(((self.frame_width * (position[0] + 1)) / self.width) + self.offset[0]) * self.scale
        y_from = np.array(((self.frame_height * position[1]) / self.height) + self.offset[1]) * self.scale
        y_to = np.array(((self.frame_width * (position[1] + 1)) / self.height) + self.offset[1]) * self.scale
        y_from, y_to, x_from, x_to = y_from.astype(int), y_to.astype(int), x_from.astype(int), x_to.astype(int)
        aoi = img[y_from:y_to, x_from:x_to] # area of interest
        return np.mean(aoi, axis=1).mean(axis=0)
