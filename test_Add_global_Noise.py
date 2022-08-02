import os
from unittest import TestCase, mock
from unittest.mock import Mock

from Add_global_Noise import createNoisyDirectory

class Test(TestCase):


    def test_create_noisy_directory_when_output_path_is_empty(self):
        #Assert
        output_dir_path = ''

        #When
        expectedResult = createNoisyDirectory(output_dir_path)

        #Then
        self.assertEqual(expectedResult, 'The output path is empty')

    def test_create_noisy_directory_when_output_path_already_exist(self):
        #Assert
        output_dir_path = './images'

        #When
        expectedResult = createNoisyDirectory(output_dir_path)

        #Then
        self.assertEqual(expectedResult, 'Noisy images directory already exist !')

    @mock.patch('os.makedirs', mock.Mock(return_value=0))
    def test_create_noisy_directory_when_creating_directory(self):
        #Assert
        output_dir_path = './test1'

        #When
        expectedResult = createNoisyDirectory(output_dir_path)

        #Then
        self.assertEqual(expectedResult, 'Noisy images directory is created !')



