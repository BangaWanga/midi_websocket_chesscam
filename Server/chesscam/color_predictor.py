import numpy as np
from typing import Tuple
import datetime
import os
import json
from pathlib import Path

# def predict_color(col_pred: ColorPrediction):
#    match col_pred:
#        case ColorPrediction.TTest:


class ColorPredictor:

    @staticmethod
    def get_complementary_color(col_rgb: tuple):
        r, g, b = col_rgb
        k = ColorPredictor.hilo(r, g, b)
        r, g, b = (255 - i for i in (r, g, b))
        return tuple(k - u for u in (r, g, b))

    def predict_color(self, col) -> Tuple[str, float]:
        pass

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)  # only difference

    @staticmethod
    def hilo(a, b, c):
        if c < b:
            b, c = c, b
        if b < a:
            a, b = b, a
        if c < b:
            b, c = c, b
        return a + c


class RangeBased(ColorPredictor):
    def __init__(self, colors=("green", "red", "blue", "yellow")):
        self.colors = colors
        self.color_data = [[] for _ in self.colors]
        self.save_file_path = "CalibrationData"
        self.init_save_folder()
        self.load_latest_save_file()
        self.avg_rgb_values = np.array([])
        self.update_rgb_averages()

    def update_rgb_averages(self):
        self.avg_rgb_values = [np.mean(c_val, axis=0) if c_val else np.array([np.nan, np.nan, np.nan])
                               for c_val in self.color_data]

    def mse(self, col):
        mse = np.array(self.avg_rgb_values)
        mse = mse - np.array(col)
        mse = np.square(mse)
        try:
            error = np.sqrt(mse) / 255
            return np.sum(error, axis=1)
        except Exception as _e:
            print("np.sqrt() failed in color prediction")
            return "None", -1.

    def predict_color(self, col, sensitivity=0.5) -> Tuple[int, float]:
        error = self.mse(col)
        if np.isnan(error).all():
            return "0", -1.
        if (error[~np.isnan(error)] > sensitivity).all():
            return "None", -1
        col_class = int(np.nanargmin(error))
        return self.color_class_to_str(col_class), float(error[col_class])

    def color_class_to_str(self, col_class: int):
        if len(self.colors) - 1 < col_class:
            raise ValueError("Color Class is not available")
        return self.colors[col_class]

    def add_sample(self, color_class: int, rgb_value: Tuple[int, int, int]):
        if color_class >= len(self.colors):
            raise ValueError("Color Class not found")
        self.color_data[color_class].append(list(rgb_value))
        # print(f"Sample added for {self.colors[color_class]}")

    def init_save_folder(self):
        if not os.path.exists(self.save_file_path):
            os.mkdir(self.save_file_path)

    def save_samples(self):
        filepath = Path(self.save_file_path).joinpath(
            f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
        config_dict = {col: self.color_data[self.colors.index(col)] for col in self.colors}
        with open(filepath, 'w') as outfile:
            outfile.write(json.dumps(config_dict))
        print("Safed samples to ", filepath)

    def load_save_file(self, filepath):
        print(f"Loading latest safe file {filepath} ")
        with open(filepath, 'r') as infile:
            config_dict = json.loads(infile.read())
        for k, v in config_dict.items():
            if k in self.colors:
                self.color_data[self.colors.index(k)] = v
                print(f"{k}: {len(v)} samples")

    def load_latest_save_file(self):  # -1 is always the lastest index
        save_files = os.listdir(self.save_file_path)
        save_files.sort()
        if save_files:
            filepath = Path(self.save_file_path).joinpath(save_files[-1])
            self.load_save_file(filepath)
#
#
# class TTest(ColorPredictor):
#     def __init__(self, colors=("red", "green", "blue")):
#         self.colors = colors
#         self._data = [np.empty((0, 3)) for _ in range(colors.__len__())]
#
#     def submit_data(self, color_rgb, label: int):
#         print(np.array(color_rgb).dtype, np.array(color_rgb).shape, self._data[label].dtype, self._data[label].shape)
#         self._data[label] = np.concatenate((self._data[label], np.array(color_rgb).reshape(1, -1)))
#
#     def check_color(self, col, crit_val) -> str:
#         _probabilities = []
#         for color_label in range(len(self._data)):
#             p = T_Test(self._data[color_label], col)
#             _probabilities.append(2 * min(p, 1 - p) * 100)
#         if any([p > crit_val for p in _probabilities]):
#             _probabilities
#         return self.colors[int(np.argmax(_probabilities))]
#
#
# def T_Test(X: np.ndarray, sample):
#     print(X.shape)
#     n_samples, _ = X.shape
#     std_deviation = np.std(X)
#     t = np.sqrt(X) * ((np.mean(X), - sample) / std_deviation)
#     s = np.random.standard_t(len(X), size=100000)
#     p = np.sum(s < t) / float(len(s))
#     print("There is a {} % probability that the paired samples stem from distributions with the same means.".format(2 * min(p, 1 - p) * 100))
#     return p
#
#
# class ColorPredictor(ColorPredictor):
#     def __init__(self, weights: tuple = ((1., 0, 0), (0., 1., 0.), (0., 0., 1.)), biases: tuple = ((0., 0., 0.)*3)):
#         self.weights = np.array(weights)
#         self.biases = np.array(biases)
#         self.classes = ("red", "green", "blue", "None")
#         # TODO: State undefined
#         # TODO: Train with picture of whole screen
#
#     def apply_weights_and_biases(self, color):
#         color = np.array(color) / 255.
#         color = (np.dot((self.weights + self.biases.reshape(3, 3)), color))
#         return color
#
#     def predict_color(self, color, confidence_threshold=0.7):
#         color = self.apply_weights_and_biases(color)
#         if np.mean(color) < confidence_threshold:
#             return self.classes[-1]     # undefined
#         color = self.softmax(color)
#         prediction = int(np.argmax(color))
#         color = self.classes[prediction]
#         return color
#

#     def train(self, color_rgb: tuple, label=0, learning_rate=0.01):
#         output = self.apply_weights_and_biases(color_rgb)
#         true_val = np.zeros(3).astype(float)
#         true_val[label] = 1.
#         error = output - true_val
#         m = 1  # n_examples
#         dW = (1.0 / m) * np.matmul(error, np.transpose(true_val))
#         db = (1.0 / m) * np.sum(error, axis=0, keepdims=True)
#         self.weights[label] = self.weights[label] - learning_rate * dW
#         self.biases[label] = self.biases[label] - learning_rate * db