from unittest import TestCase

from pathlib import Path
import numpy as np
import cv2
from Server.chesscam.image_processing import standardize_position


class TestImageStandardization(TestCase):
    def test_output_types(self):
        # image that should yield a result
        good_impath = str(Path(__file__).parent / 'resources' / 'synthetisch' / 'board.png')
        good_img = cv2.imread(good_impath)
        self.assertIsInstance(standardize_position(good_img), np.ndarray)

        # image that should yield None
        bad_impath = str(Path(__file__).parent / 'resources' / 'synthetisch' / 'no_board.jpg')
        bad_img = cv2.imread(bad_impath)
        self.assertIsNone(standardize_position(bad_img))

    def test_recognition(self):
        # images that should be successfully recognized
        valid_paths = (Path(__file__).parent / 'resources' / 'fotos').glob('valid_*')
        for impath in valid_paths:
            img_ = cv2.imread(str(impath))
            self.assertIsInstance(standardize_position(img_), np.ndarray)

        # images that should be rejected
        invalid_paths = [Path(__file__).parent / 'resources' / 'synthetisch' / 'no_board.jpg']
        for impath in invalid_paths:
            img_ = cv2.imread(str(impath))
            self.assertIsNone(standardize_position(img_))
