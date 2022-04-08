import cv2
import numpy as np
import numpy.ma as ma


def get_(edge0, edge1, edge2, edge3):

    pass


class Overlay:
    def __init__(self, frame_shape):
        self.frame_shape = frame_shape
        print(f"Frame shape: {self.frame_shape}")
        self.grid = Grid(8, 8)

    def draw_line(self, img, start=(0, 0), end=(100, 100), line_thickness=2, col=(0, 255, 0)):
        img = img.copy()
        import cv2
        cv2.line(img, start, end, col, thickness=line_thickness)
        return img

    def draw_rectangle(self, img, pts1=(0, 0), pts2=(100, 100)):
        cv2.rectangle(img, pts1, pts2,color=(0, 0, 0), thickness=3)

    def draw_grid(self, img, offset=(0, 0)):
        sp, ep = self.grid.get_line_coordinates(width=int(self.frame_shape[1]), height=int(self.frame_shape[0]))
        print(sp, ep)
        for i in range(len(sp)):
            start_point = (sp[i][0] + offset[0], sp[i][1] + offset[1])
            end_point = (ep[i][0] + offset[0], ep[i][1] + offset[1])
            img = self.draw_line(img, start_point, end_point)
        return img


class Grid:
    def __init__(self, width: int, height: int, n_classes: int = 3):
        self.grid_vectors = [[{(x, y): None} for x in range(9)] for y in range(9)]
        self.grid = [[{(x, y): None} for x in range(9)] for y in range(9)]
        self.width = width
        self.height = height

    def make_grid_position_iterator(self):
        return [[(i, j) for j in range(8)]for i in range(8)].__iter__()

    @staticmethod
    def get_line_points(width: int, height: int):
        start_positions_horizontal = np.column_stack(((np.zeros(width, dtype=int),
                                                       (np.arange(width) * height / (width - 1)).astype(int))))
        start_positions_vertical = np.column_stack(((np.arange(height) * width / (height - 1)).astype(int),
                                                    np.zeros(height, dtype=int)))

        end_positions_horizontal = np.column_stack((np.ones(width, dtype=int) * width,
                                                    (np.arange(width) * height / (width - 1)).astype(int)))
        end_positions_vertical = np.column_stack(((np.arange(height) * width / (height - 1)).astype(int),
                                                  np.ones(height, dtype=int) * height))


    @staticmethod
    def get_line_coordinates(width: int, height: int, scale=1.):
        lines = 9

        sp = np.vstack((start_positions_horizontal, start_positions_vertical))
        ep = np.vstack((end_positions_horizontal, end_positions_vertical))

        sp = (sp * scale).astype(int)
        ep = (ep * scale).astype(int)

        sp = [tuple(pos) for pos in sp]
        ep = [tuple(pos) for pos in ep]
        return sp, ep

    def get_rect_points(self, x: int, y: int):
        field_width = self.width / 8
        field_height = self.height / 8
        return (int(x * field_width), y * field_height), \
               (int(x * field_width + field_width), int(x * field_height + field_height))

    def get_pos_vectors(self, x: int, y: int):
        support_vectors = [(x, y), (x, y + 1), (x + 1, y)]
        direction_vectors = [(0, 8), (8, 0)]
        ret = []
        for i in range(2):
            for j in range(2):
                print()
                ret.append((support_vectors[i], direction_vectors[j]))
        return ret

    """
    def create_mask_for_field(self, image, x: int, y: int):
        mask = ma.masked_where(a <= 2, a)
        is_in_field = lambda edge0, edge1, edge2, edge3: # the condition is, if you cast a vector to the edges it can't cross outer connections between the edges
        X = np.vstack(np.arange(8) for _ in range(8))
        X = np.zeros((8, 8))
        return ma.masked_array(image, mask=[0, 0, 0, 1, 0])
    """


