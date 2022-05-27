import numpy as np
from typing import Tuple, Optional, List
import datetime
import os
import json
from pathlib import Path


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


class NearestNeighbour(ColorPredictor):
    def __init__(self, colors=("null", "green", "red", "blue", "yellow")):
        self.colors = colors
        self.color_data = [[] for _ in self.colors]
        self.save_file_path = "CalibrationData"
        self.init_save_folder()
        # self.load_latest_save_file()
        self.avg_rgb_values = np.array([])
        self.std_deviance = np.zeros(shape=(len(self.colors), 3), dtype=float)  # variance for each channel
        self.update_rgb_averages()

    def update_rgb_averages(self):
        self.avg_rgb_values = np.array([np.mean(c_val, axis=0) if c_val else np.array([np.nan, np.nan, np.nan])
                                        for c_val in self.color_data])
        print(self.avg_rgb_values)
        for idx in range(len(self.color_data)):   # we don't know how many samples are collected for each color, so we use for-loop
            color_samples = self.color_data[idx]
            if not color_samples:
                self.std_deviance[idx] = np.array([0., 0., 0.]) # ToDo: Is this correct?
            else:
                cs = np.array(color_samples)
                # cs = np.convolve(np.ones(shape=cs.shape), cs/len(cs), mode="full")
                self.std_deviance[idx] = np.sqrt(
                    np.sum(np.square(cs - self.avg_rgb_values[idx]), axis=0)
                )

    def calculate_error(self, color_value) -> Optional[np.array]:
        mse = self.avg_rgb_values - np.array(color_value)
        mse = np.square(mse)
        try:
            mse = np.sqrt(mse)
            error = np.sum(mse / self.std_deviance, axis=1)
            return error
        except Exception as _e:
            print("np.sqrt() failed in color prediction")
            return None

    def predict_color(self, col, sensitivity=.05) -> Tuple[Optional[int], float]:
        # ToDo: bigger numpy array
        error = self.calculate_error(col)
        if np.isnan(col).all():
            return "~COL", -1.
        if np.isnan(error).all():
            return "MISS", -1.
        if (error[~np.isnan(error)] > sensitivity).all():
            return None, -1
        col_class = int(np.nanargmin(error))

        return col_class, float(error[col_class])

    def color_class_to_str(self, col_class: int):
        if len(self.colors) - 1 < col_class:
            raise ValueError("Color Class is not available")
        return self.colors[col_class]

    def add_samples(self, color_classes: List[int], rgb_values: List[Tuple[int, int, int]], write: bool = False) -> bool:
        for i in range(len(color_classes)):
            color_class = color_classes[i]
            rgb_value = rgb_values[i]
            self.color_data[color_class].append(list(rgb_value))
        if write:
            return self.save_samples()
        else:
            return True

    def init_save_folder(self):
        if not os.path.exists(self.save_file_path):
            os.mkdir(self.save_file_path)

    def save_samples(self) -> bool:
        filepath = Path(self.save_file_path).joinpath(
            f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json")
        config_dict = {col: self.color_data[self.colors.index(col)] for col in self.colors}
        try:
            with open(filepath, 'w') as outfile:
                outfile.write(json.dumps(config_dict))
                return True
        except:
            return False

    def load_save_file(self, filepath):
        print(f"Loading latest safe file {filepath} ")
        with open(filepath, 'r') as infile:
            config_dict = json.loads(infile.read())
        for k, v in config_dict.items():
            if k in self.colors:
                self.color_data[self.colors.index(k)] = v
                print(f"{k}: {len(v)} samples")
        return True

    def load_latest_save_file(self):  # -1 is always the lastest index
        save_files = os.listdir(self.save_file_path)
        save_files.sort()
        if save_files:
            filepath = Path(self.save_file_path).joinpath(save_files[-1])
            try:
                return self.load_save_file(filepath)
            except:
                return False
        else:
            return False
