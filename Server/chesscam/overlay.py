import numpy as np
import math
import itertools
import cv2
from typing import Tuple
import random
from enum import Enum
from color_predictor import ColorPredictor, RangeBased


class DisplayOption(Enum):
    MatchingCircles = 0
    RandomColors = 1
    Classes = 2
    ClassesRangeBased = 3


class Overlay:
    def __init__(self, frame_shape, width: int = 8, height: int = 8, offset: Tuple[int, int] = (0, 0),
                 scale: float = 1.):
        self.frame_shape = frame_shape
        print(frame_shape)
        self.width = width
        self.height = height
        self.frame_width = int(self.frame_shape[1])
        self.frame_height = int(self.frame_shape[0])
        self.offset = offset
        self.scale = scale
        self.grid = {}
        self.update_grid(False)
        self.display_option = DisplayOption.ClassesRangeBased
        self.color_predictor = ColorPredictor()
        self.range_based = RangeBased()

    def calculate_field_width(self):
        return int((self.frame_width * self.scale) / self.width)

    def calculate_field_height(self):
        return int((self.frame_height * self.scale) / self.height)

    def train_all_fields(self, img, color: int):
        #assert color in [1, 2, 3]
        for x in range(self.width):
            for y in range(self.height):
                center_point = self.grid[(x, y)]["center_point"]
                X = self.scan_square(img, center_point) # X is a rgb value
                self.train_color(X, color, )

    def train_color(self, img, color: int, learn_rate=0.1):
        self.color_predictor.train(img, color, learn_rate)

    def add_sample(self, img, field_pos: Tuple[int, int], color_label: int):
        center_point = self.grid[field_pos]["center_point"]
        X = self.scan_square(img, center_point)  # X is a rgb value
        self.range_based.submit_data(X, color_label)

    def update_grid(self, ignore_colors: bool):
        grid = {}
        for i in range(self.height):
            for j in range(self.width):

                random_col = tuple(random.randint(0, 255) for _ in range(3))
                edge0, edge1, edge2, edge3 = self.get_pixel_edges(j, i, self.offset, self.scale)
                line_coordinates = self.get_start_and_endpoints_from_edges(edge0, edge1, edge2, edge3)
                center_point = self.find_center_of_square(edge0, edge1, edge2, edge3)
                if ignore_colors: # Get old colors to new positions
                    if (i, j) not in self.grid:
                        raise ValueError("Grid is not initalized but ignore colors is False")
                    grid[(i, j)] = {"line_coordinates": line_coordinates,
                                    "center_point": center_point,
                                    "color": self.grid[(i, j)]["color"]}
                else:
                    grid.update(
                        {(i, j):
                             {"line_coordinates": line_coordinates, "center_point": center_point, "color": random_col}}
                        )
        self.grid = grid

    def scroll_display_option(self):
        options = [d for d in DisplayOption]
        self.display_option = DisplayOption(options[(options.index(self.display_option) + 1) % len(options)])   # sorry

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

    def write_text(self, img, text: str, start_pos=(0, 0), fontScale: int = 1, col=(255, 255, 255), thickness=1,
                   lineType=1):
        img = img.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        start_pos = start_pos
        fontScale = fontScale
        fontColor = col
        thickness =thickness
        lineType = lineType
        cv2.putText(img, text,
                    start_pos,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
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

    def change_drawing_options(self, offset: Tuple[int, int] = (0, 0), scale: float = 1., ignore_colors: bool = True):
        self.offset = offset
        self.scale = scale
        self.update_grid(ignore_colors=ignore_colors)
        # ToDo: Take care that we cant exit screen

    def draw_grid(self, img):
        for k, v in self.grid.items():
            line_coordinates = v["line_coordinates"]
            for line in line_coordinates:
                startpoint, endpoint = line
                img = self.draw_line(img, startpoint, endpoint, col=v["color"])
            # draw circle in center:
            match self.display_option:
                case DisplayOption.MatchingCircles:
                    img = self.draw_matching_circles(img, v["center_point"])
                case DisplayOption.RandomColors:
                    img = self.draw_circle(img, v["center_point"], col=v["color"])
                case DisplayOption.Classes:
                    img = self.draw_classes(img, v["center_point"])
                case DisplayOption.ClassesRangeBased:
                    img = self.draw_classes_range_based(img, v["center_point"])
        return img

    def draw_classes_range_based(self, img, center_point):
        color = self.scan_square(img, center_point)
        color_class = self.range_based.check_color(color)
        print("Color Class: ", color_class)
        col_complementary = self.color_predictor.get_complementary_color(color)
        text_offset = (- int(self.calculate_field_width() / 2), - int(self.calculate_field_height() / 2))
        center_point = (center_point[0] + text_offset[0], center_point[1] + text_offset[1])
        img = self.write_text(img, f"{color_class}", start_pos=center_point, col=col_complementary)
        return img

    def draw_classes(self, img, center_point):
        color = self.scan_square(img, center_point)
        color_class = self.color_predictor.predict_color(color)
        col_complementary = self.color_predictor.get_complementary_color(color)
        text_offset = (- int(self.calculate_field_width() / 2), - int(self.calculate_field_height() / 2))
        center_point = (center_point[0] + text_offset[0], center_point[1] + text_offset[1])
        img = self.write_text(img, f"{color_class}", start_pos=center_point, col=col_complementary)

        return img

    def draw_matching_circles(self, img, center_point: (0, 0), debug=False):
        col = self.scan_square(img, center_point)
        img = self.draw_circle(img, center_point, col=col, radius=20)
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
    def get_region(img, square_center: Tuple[int, int], scan_width: int = 2):
        s0 = square_center[0] - int(scan_width / 2)
        e0 = square_center[0] + int(scan_width / 2)
        s1 = square_center[1] - int(scan_width / 2)
        e1 = square_center[1] + int(scan_width / 2)
        region = img[s1:e1, s0:e0]
        return region

    @staticmethod
    def scan_square(img, square_center: Tuple[int, int], scan_width: int = 2, debug=False):
        region = Overlay.get_region(img, square_center, scan_width)
        color_value = np.mean(region, axis=1).mean(axis=0)
        if debug:
            print(f"colorValue: Red {color_value[0]} Green {color_value[1]} Blue {color_value[2]}", )
        return color_value

    def aquire_data(self, img, color: int):
        #assert color in [1, 2, 3]
        for x in range(self.width):
            for y in range(self.height):
                center_point = self.grid[(x, y)]["center_point"]
                X = self.scan_square(img, center_point) # X is a rgb value

