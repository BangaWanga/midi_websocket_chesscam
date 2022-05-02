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


class Overlay:
    def __init__(self, frame_shape, width: int = 8, height: int = 8, offset: Tuple[int, int] = (0, 0),
                 scale: float = 1.):
        self.width = width
        self.height = height
        self.frame_height, self.frame_width = frame_shape
        self.offset = offset
        self.scale = scale
        self.grid = {}
        self.update_grid(False)
        self.display_option = DisplayOption.Calibrate
        self.color_predictor = RangeBased()
        #   self.range_based = TTest()
        self.cursor_field = (0, 0)  # ToDo: Less variables for cursor
        self.cursor = np.array([0., 0.])
        self.cursor_absolute = (0., 0.)
        self.selected_color = None
        self.calibrate_field = False  # If this value is tuple[int, int], the selected field is calibrated with selected color

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
        img = img.copy()
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

    def change_drawing_options(self, offset: Tuple[int, int] = (0, 0), scale: float = 1., ignore_colors: bool = True):
        self.offset = offset
        self.scale = scale
        self.update_grid(ignore_colors=ignore_colors)
        # ToDo: Take care that we cant exit screen

    def draw_cursor(self, img):
        if self.cursor_field is None:
            return img
        img = img.copy()
        left_upper_corner = self.get_rect_start_position(self.cursor_field)
        right_lower_corner = int(left_upper_corner[0] + self.rect_width), int(left_upper_corner[1] + self.rect_height)
        img = self.draw_rectangle(img, left_upper_corner, right_lower_corner)
        return img

    def draw_grid(self, img, grid_color=(100, 50, 50)):  # ToDo: Make differentiation between display modes earlier
        for k, v in self.grid.items():
            left_upper_corner = self.get_rect_start_position(k)
            right_lower_corner = int(left_upper_corner[0] + self.rect_width), int(
                left_upper_corner[1] + self.rect_height)
            img = self.draw_rectangle(img, left_upper_corner, right_lower_corner, col=grid_color)

            #            line_coordinates = v["line_coordinates"]
            #            for line in line_coordinates:
            #                startpoint, endpoint = line
            #                img = self.draw_line(img, startpoint, endpoint, col=v["color"])
            # draw circle in center:
            match self.display_option:
                case DisplayOption.MatchingCircles:
                    img = self.draw_matching_circles(img, v["center_point"])
                case DisplayOption.RandomColors:
                    img = self.draw_circle(img, v["center_point"], col=v["color"])
                case DisplayOption.Calibrate:
                    img = self.calibrate(img, k, v)
                    img = self.draw_classes(img, v["center_point"])
                case DisplayOption.Classes:
                    img = self.draw_classes(img, v["center_point"])
                case DisplayOption.ClassesRangeBased:
                    img = self.draw_classes_range_based(img, v["center_point"], v["edges"], k)
        img = self.draw_cursor(img)
        return img

    def calibrate(self, img, position: tuple, _field_info: dict):
        if self.selected_color is None or not self.calibrate_field:
            return img
        color = self.scan_square2(img, position)
        if position == self.cursor_field:
            print("Adding sample ", color, " for field ", position, " and color ", self.selected_color, self.color_predictor.colors[self.selected_color])
            self.color_predictor.add_sample(self.selected_color, color)
            self.calibrate_field = False    # ToDo: Maybe add more samples right away?
            # self.color_predictor.add_sample(self.selected_color, color)
        return img

    def draw_classes(self, img, center_point):  # TODO: If Calibrate mode, draw error to UI somehow
        color = self.scan_square(img, center_point)
        color_class, error = self.color_predictor.predict_color(color)
        col_complementary = self.color_predictor.get_complementary_color(color)
        text_offset = (- int(self.rect_width / 2), - int(self.rect_height / 2))
        center_point = (center_point[0] + text_offset[0], center_point[1] + text_offset[1])

        # error2color = cv2.cvtColor(np.array(error * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB).reshape(3)
        # error2color = tuple(np.array(error2color) / 255)
        # x = f"{str(_error)}"

        img = self.write_text(img, f"{color_class}", start_pos=center_point, font_color=col_complementary)

        return img

    def draw_matching_circles(self, img, center_point: (0, 0), debug=False):
        col = self.scan_square(img, center_point)
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

    def update_cursor(self, sensitivity_threshold=.1, movement_speed=0.05):
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
        # print(f"axis {axis} value {value}")
        self.update_cursor()

    @staticmethod
    def scan_square(img, square_center: Tuple[int, int], scan_width: int = 2, debug=False):
        region = Overlay.get_region(img, square_center, scan_width)
        color_value = np.mean(region, axis=1).mean(axis=0)
        if debug:
            print(f"colorValue: Red {color_value[0]} Green {color_value[1]} Blue {color_value[2]}", )
        return color_value

    def scan_square2(self, img, position: Tuple[int, int]):
        x_from = np.array((self.frame_width * position[0] / self.width) + self.offset[0]) * self.scale
        x_to = np.array(((self.frame_width * (position[0] + 1)) / self.width) + self.offset[0]) * self.scale
        y_from = np.array(((self.frame_height * position[1]) / self.height) + self.offset[1]) * self.scale
        y_to = np.array(((self.frame_width * (position[1] + 1)) / self.height) + self.offset[1]) * self.scale
        y_from, y_to, x_from, x_to = y_from.astype(int), y_to.astype(int), x_from.astype(int), x_to.astype(int)
        aoi = img[y_from:y_to, x_from:x_to] # area of interest
        return np.mean(aoi, axis=1).mean(axis=0)

    def aquire_data(self, img, color: int):
        # assert color in [1, 2, 3]
        for x in range(self.width):
            for y in range(self.height):
                center_point = self.grid[(x, y)]["center_point"]
                X = self.scan_square(img, center_point)  # X is a rgb value
