import numpy as np
import colorsys


class RangeBased:
    def __init__(self, colors=("red", "green", "blue")):
        self.colorBoundaries = [
            [np.array([50, 56, 50]), np.array([255, 56, 50])],  # red, lower boarder, upper boarder
            [np.array([20, 200, 50]), np.array([20, 255, 20])],   # green, ...
            [np.array([0, 56, 50]), np.array([50, 88, 220])]    # blue, ...
        ]
        self.colors = colors
        self._data = [np.empty((0, 3)) for _ in range(colors.__len__())]

    def submit_data(self, color_rgb, label: int):
        print(np.array(color_rgb).dtype, np.array(color_rgb).shape, self._data[label].dtype, self._data[label].shape)
        self._data[label] = np.concatenate((self._data[label], np.array(color_rgb).reshape(1, -1)))

    def check_color(self, col, crit_val) -> str:
        _probabilities = []
        for color_label in range(len(self._data)):
            p = T_Test(self._data[color_label], col)
            _probabilities.append(2 * min(p, 1 - p) * 100)
        if any([p > crit_val for p in _probabilities]):
            _probabilities
        return self.colors[int(np.argmax(_probabilities))]


def T_Test(X: np.ndarray, sample):
    print(X.shape)
    n_samples, _ = X.shape
    std_deviation = np.std(X)
    t = np.sqrt(X) * ((np.mean(X), - sample) / std_deviation)
    s = np.random.standard_t(len(X), size=100000)
    p = np.sum(s < t) / float(len(s))
    print("There is a {} % probability that the paired samples stem from distributions with the same means.".format(2 * min(p, 1 - p) * 100))
    return p


class ColorPredictor:
    def __init__(self, weights: tuple = ((1., 0, 0), (0., 1., 0.), (0., 0., 1.)), biases: tuple = ((0., 0., 0.)*3)):
        self.weights = np.array(weights)
        self.biases = np.array(biases)
        self.classes = ("red", "green", "blue", "None")
        # TODO: State undefined
        # TODO: Train with picture of whole screen

    def apply_weights_and_biases(self, color):
        color = np.array(color) / 255.
        color = (np.dot((self.weights + self.biases.reshape(3, 3)), color))
        return color

    def predict_color(self, color, confidence_threshold=0.7):
        color = self.apply_weights_and_biases(color)
        if np.mean(color) < confidence_threshold:
            return self.classes[-1]     # undefined
        color = self.softmax(color)
        prediction = int(np.argmax(color))
        color = self.classes[prediction]
        return color

    @staticmethod
    def hilo(a, b, c):
        if c < b: b, c = c, b
        if b < a: a, b = b, a
        if c < b: b, c = c, b
        return a + c

    @staticmethod
    def complement(r, g, b):
        k = ColorPredictor.hilo(r, g, b)
        return tuple(k - u for u in (r, g, b))

    @staticmethod
    def get_complementary_color(col):
        #col = (i / 255.0 for i in col)
        #hlsval = colorsys.rgb_to_hls(*col)
        #print(f"hlsval: {hlsval}")
        #rgbval = colorsys.hls_to_rgb(*hlsval)
        return ColorPredictor.complement(*col)

    @staticmethod
    def softmax(x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)  # only difference

    def train(self, color_rgb: tuple, label=0, learning_rate=0.01):
        output = self.apply_weights_and_biases(color_rgb)
        true_val = np.zeros(3).astype(float)
        true_val[label] = 1.
        error = output - true_val
        m = 1  # n_examples
        dW = (1.0 / m) * np.matmul(error, np.transpose(true_val))
        db = (1.0 / m) * np.sum(error, axis=0, keepdims=True)
        self.weights[label] = self.weights[label] - learning_rate * dW
        self.biases[label] = self.biases[label] - learning_rate * db