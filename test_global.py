import unittest
from util import Read_Image_to_numpy_arr
import numpy as np


class GlobalTestCase(unittest.TestCase):

    def test_image_values(self):
        # assert
        image_path = "image_1_mask.png"
        # when
        image = Read_Image_to_numpy_arr(image_path)
        max_value = np.amax(image)
        min_value = np.amin(image)
        # then
        self.assertEqual(min_value, 0)  # add assertion here
        self.assertEqual(max_value, 255)

    def test_image_size(self):
        # assert
        image_path = "image_1_mask.png"
        # when
        image = Read_Image_to_numpy_arr(image_path)
        #then
        self.assertEqual(image.shape, (2048, 2048, 3))


if __name__ == '__main__':
    unittest.main()
