import unittest
from util import Read_Image_to_numpy_arr
import numpy as np
from unittest.mock import Mock
from util import create_directory
from unittest import TestCase, mock


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


    def test_create_directory_when_output_path_is_empty(self):
        #Assert
        output_dir_path = ''

        #When
        expectedResult = create_directory(output_dir_path)

        #Then
        self.assertEqual(expectedResult, 'The output path is empty')

    def test_create_directory_when_output_path_already_exist(self):
        #Assert
        output_dir_path = './images'

        #When
        expectedResult = create_directory(output_dir_path)

        #Then
        self.assertEqual(expectedResult, 'Noisy images directory already exist !')

    @mock.patch('os.makedirs', mock.Mock(return_value=0))
    def test_create_directory_when_creating_directory(self):
        #Assert
        output_dir_path = './test1'

        #When
        expectedResult = create_directory(output_dir_path)

        #Then
        self.assertEqual(expectedResult, 'Noisy images directory is created !')






if __name__ == '__main__':
    unittest.main()
